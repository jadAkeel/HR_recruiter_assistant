import { useState, useRef } from 'react';
import api from '../../api/client';
import { Upload, CheckCircle, XCircle, Loader, FileText } from 'lucide-react';
import { getApiErrorMessage, getApiStatus } from '../../utils/errors';

const MAX_NETWORK_RETRIES = 3;

const sleep = (milliseconds: number) => new Promise((resolve) => setTimeout(resolve, milliseconds));

const isTransientError = (error: unknown) => {
  const status = getApiStatus(error);
  return status === undefined || status === 408 || status === 425 || status === 429 || status >= 500;
};

async function withNetworkRetry<T>(operation: () => Promise<T>): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt < MAX_NETWORK_RETRIES; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (!isTransientError(error) || attempt === MAX_NETWORK_RETRIES - 1) throw error;
      await sleep(500 * (attempt + 1));
    }
  }
  throw lastError instanceof Error ? lastError : new Error('Network request failed');
}

export default function BulkUpload() {
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<Record<string, { status: string; name?: string; error?: string }>>({});
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const pollTasks = async (tasks: Array<{ task_id: string; filename: string }>) => {
    const pending = new Map(tasks.map((task) => [task.task_id, task.filename]));
    for (let attempt = 0; attempt < 180; attempt += 1) {
      const taskIds = Array.from(pending.keys());
      if (taskIds.length === 0) return;

      const { data } = await withNetworkRetry(() => api.get<Record<string, {
        status: string;
        full_name?: string;
        error?: string;
      }>>('/candidates/async/bulk', {
        params: { task_ids: taskIds },
        paramsSerializer: () => taskIds.map((taskId) => `task_ids=${encodeURIComponent(taskId)}`).join('&'),
        timeout: 30_000,
      }));

      for (const [taskId, filename] of pending) {
        const result = data[taskId];
        if (!result || result.status === 'pending') continue;
        if (result.status === 'failed') {
          setResults((prev) => ({
            ...prev,
            [filename]: { status: 'fail', error: result.error || 'Processing failed' },
          }));
        } else if (result.status === 'completed') {
          setResults((prev) => ({
            ...prev,
            [filename]: { status: 'ok', name: result.full_name || 'Unknown' },
          }));
        } else {
          continue;
        }
        pending.delete(taskId);
      }

      await sleep(2000);
    }
    for (const filename of pending.values()) {
      setResults((prev) => ({
        ...prev,
        [filename]: { status: 'fail', error: 'Processing timed out' },
      }));
    }
  };

  const uploadAll = async () => {
    if (files.length === 0) return;
    setUploading(true);
    setResults(Object.fromEntries(files.map((file) => [file.name, { status: 'uploading' }])));
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append('files', file, file.name));
      const { data } = await withNetworkRetry(() => api.post('/candidates/async/bulk', formData, {
        params: { use_llm: true },
        timeout: 120_000,
      }));
      const queuedTasks: Array<{ task_id: string; filename: string }> = [];
      for (const task of data.tasks as Array<{
        task_id?: string;
        filename: string;
        status: string;
        error?: string;
      }>) {
        if (task.status === 'queued' && task.task_id) {
          queuedTasks.push({ task_id: task.task_id, filename: task.filename });
          setResults((prev) => ({ ...prev, [task.filename]: { status: 'queued' } }));
        } else {
          setResults((prev) => ({
            ...prev,
            [task.filename]: { status: 'fail', error: task.error || 'Upload failed' },
          }));
        }
      }
      await pollTasks(queuedTasks);
    } catch (err: unknown) {
      const message = getApiErrorMessage(err, 'Bulk upload failed');
      setResults((prev) => Object.fromEntries(
        Object.entries(prev).map(([filename, result]) => (
          result.status === 'ok' || result.status === 'fail'
            ? [filename, result]
            : [filename, { status: 'fail', error: message }]
        )),
      ));
    } finally {
      setUploading(false);
    }
  };

  const done = Object.values(results).filter((r) => r.status === 'ok').length;
  const failed = Object.values(results).filter((r) => r.status === 'fail').length;

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Bulk Upload CVs</h1>

      <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (e.dataTransfer.files) setFiles(Array.from(e.dataTransfer.files));
          }}>
          <Upload className="w-12 h-12 mx-auto mb-3 text-blue-500" />
          <p className="text-gray-600 mb-2">Drag & drop CV files here, or click to select</p>
          <p className="text-xs text-gray-400 mb-4">PDF, DOCX, TXT — up to 15MB each</p>
          <input ref={inputRef} type="file" multiple accept=".pdf,.docx,.txt"
            onChange={handleSelect} className="hidden" />
          <button onClick={() => inputRef.current?.click()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            Select Files
          </button>
        </div>

        {files.length > 0 && (
          <div className="mt-4">
            <p className="text-sm text-gray-600 mb-2">{files.length} file(s) selected</p>
            <div className="flex gap-2">
              <button onClick={uploadAll} disabled={uploading}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50">
                {uploading ? 'Uploading...' : `Upload All (${files.length})`}
              </button>
              {!uploading && (
                <button onClick={() => { setFiles([]); setResults({}); }}
                  className="px-4 py-2 border rounded-lg hover:bg-gray-50">Clear</button>
              )}
            </div>
          </div>
        )}

        {(done > 0 || failed > 0) && (
          <div className="mt-3 flex gap-4 text-sm">
            <span className="text-green-600">✓ {done} success</span>
            {failed > 0 && <span className="text-red-600">✗ {failed} failed</span>}
          </div>
        )}
      </div>

      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((file) => {
            const res = results[file.name];
            return (
              <div key={file.name} className="bg-white rounded-lg border p-3 flex items-center gap-3">
                {!res ? (
                  <FileText className="w-5 h-5 text-gray-400" />
                ) : res.status === 'uploading' || res.status === 'queued' ? (
                  <Loader className="w-5 h-5 text-blue-500 animate-spin" />
                ) : res.status === 'ok' ? (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-500" />
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">{file.name}</p>
                  {res?.status === 'ok' && (
                    <p className="text-xs text-green-600">{res.name}</p>
                  )}
                  {(res?.status === 'uploading' || res?.status === 'queued') && (
                    <p className="text-xs text-blue-500">{res.status === 'uploading' ? 'Uploading...' : 'Queued'}</p>
                  )}
                  {res?.status === 'fail' && (
                    <p className="text-xs text-red-500">{res.error}</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
