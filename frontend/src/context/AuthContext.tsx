import { useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import api, { warmApi } from '../api/client';
import { AuthContext } from './auth';
import type { User } from './auth';

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(() => Boolean(localStorage.getItem('access_token')));

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (!token) return;

    let active = true;
    let retryTimer: number | undefined;

    const restoreSession = async () => {
      try {
        await warmApi();
        const { data } = await api.get<User>('/auth/me');
        if (active) {
          setUser(data);
          setLoading(false);
        }
      } catch {
        if (!active) return;
        if (!localStorage.getItem('access_token')) {
          setLoading(false);
          return;
        }
        retryTimer = window.setTimeout(() => void restoreSession(), 2_000);
      }
    };

    void restoreSession();

    return () => {
      active = false;
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
    };
  }, []);

  const login = async (email: string, password: string) => {
    await warmApi();
    const { data } = await api.post('/auth/login', { email, password });
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    const { data: me } = await api.get<User>('/auth/me');
    setUser(me);
    return me;
  };

  const register = async (email: string, password: string, full_name: string) => {
    await warmApi();
    await api.post('/auth/register', { email, password, full_name });
    return login(email, password);
  };

  const logout = () => {
    localStorage.clear();
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
}
