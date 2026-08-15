import { useState, useRef } from 'react';
import api from '../../api/client';
import { Upload, CheckCircle, Loader2, AlertCircle, Sparkles, FileText } from 'lucide-react';
import type { CandidateUploadResult } from '../../types/api';
import { getApiErrorMessage } from '../../utils/errors';

export default function UploadCV() {
  const [result, setResult] = useState<CandidateUploadResult | null>(null);
  const [error, setError] = useState('');
  const [status, setStatus] = useState<'idle' | 'uploading' | 'done' | 'error'>('idle');
  const [useLlm, setUseLlm] = useState(false);
  const [fileName, setFileName] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    if (!file) return;
    setFileName(file.name);
    setError('');
    setStatus('uploading');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const { data } = await api.post('/candidates', formData, {
        params: { use_llm: useLlm },
      });
      setResult(data);
      setStatus('done');
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, 'Upload failed'));
      setStatus('error');
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      void handleFile(file);
    }
  };

  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Upload Your CV</h1>

      {status === 'idle' && (
        <div className="bg-white rounded-xl shadow-sm border p-8 space-y-6">
          <div
            className="border-2 border-dashed border-blue-200 rounded-xl p-10 text-center hover:border-blue-500 hover:bg-blue-50/30 transition-all cursor-pointer"
            onClick={triggerFileInput}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const droppedFile = e.dataTransfer.files?.[0];
              if (droppedFile) void handleFile(droppedFile);
            }}
          >
            <Upload className="w-16 h-16 mx-auto mb-4 text-blue-500" />
            <p className="text-lg font-semibold text-gray-800 mb-1">
              Drag & drop your CV here, or click to browse
            </p>
            <p className="text-sm text-gray-500 mb-4">
              Supported formats: PDF, DOCX, TXT (up to 15MB)
            </p>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                triggerFileInput();
              }}
              className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 shadow transition-colors"
            >
              <FileText className="w-5 h-5" />
              Choose File
            </button>
          </div>

          <div className="flex items-center justify-between pt-2 border-t text-sm text-gray-600">
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={useLlm}
                onChange={(e) => setUseLlm(e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-amber-500" />
                Enable Deep AI analysis (Ollama)
              </span>
            </label>
            <span className="text-xs text-gray-400">
              {useLlm ? 'Slower (~30s) detailed reasoning' : 'Instant (<1s) ESCO parsing'}
            </span>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            onChange={handleInputChange}
            className="hidden"
          />
        </div>
      )}

      {status === 'uploading' && (
        <div className="bg-white rounded-xl shadow-sm border p-12 text-center space-y-4">
          <Loader2 className="w-14 h-14 mx-auto text-blue-600 animate-spin" />
          <div>
            <p className="text-lg font-semibold text-gray-900">
              {useLlm ? 'Analyzing CV with AI...' : 'Parsing and indexing CV...'}
            </p>
            <p className="text-sm text-gray-500 mt-1">{fileName}</p>
          </div>
          <p className="text-xs text-gray-400">
            Extracting skills, experience, and contact details
          </p>
        </div>
      )}

      {status === 'done' && result && (
        <div className="bg-white rounded-xl shadow-sm border p-8 space-y-6">
          <div className="flex items-center gap-3 text-green-600 border-b pb-4">
            <CheckCircle className="w-7 h-7" />
            <div>
              <h2 className="text-lg font-bold">CV Processed Successfully</h2>
              <p className="text-xs text-green-700">Candidate added to the hiring database</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Full Name</p>
              <p className="text-base font-medium text-gray-900 mt-1">{result.full_name || 'Not detected'}</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Email</p>
              <p className="text-base font-medium text-gray-900 mt-1">{result.email || 'Not detected'}</p>
            </div>
          </div>

          {result.skills && result.skills.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                Extracted Skills ({result.skills.length})
              </p>
              <div className="flex gap-2 flex-wrap">
                {result.skills.map((s: string) => (
                  <span
                    key={s}
                    className="px-2.5 py-1 bg-blue-50 text-blue-700 border border-blue-100 rounded-md text-xs font-medium"
                  >
                    {s}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="pt-4 border-t flex items-center justify-between">
            <button
              onClick={() => {
                setResult(null);
                setStatus('idle');
              }}
              className="text-sm font-medium text-blue-600 hover:text-blue-800"
            >
              ← Upload another CV
            </button>
            <a
              href="/candidates"
              className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700"
            >
              View in Candidates List
            </a>
          </div>
        </div>
      )}

      {status === 'error' && (
        <div className="bg-white rounded-xl shadow-sm border p-10 text-center space-y-4">
          <AlertCircle className="w-14 h-14 mx-auto text-red-500" />
          <h2 className="text-lg font-bold text-gray-900">Upload Failed</h2>
          <p className="text-sm text-red-600 max-w-md mx-auto">{error || 'Could not process the uploaded file'}</p>
          <button
            onClick={() => {
              setError('');
              setStatus('idle');
            }}
            className="px-6 py-2.5 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors shadow"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
}
