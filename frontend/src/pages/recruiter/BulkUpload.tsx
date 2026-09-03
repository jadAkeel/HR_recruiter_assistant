import { useState, useRef } from 'react';
import api from '../../api/client';
import {
  Upload,
  CheckCircle2,
  XCircle,
  Loader2,
  FileText,
  RefreshCw,
  Square,
  Clock,
  Sparkles,
  AlertTriangle,
} from 'lucide-react';
import { getApiErrorMessage, getApiStatus } from '../../utils/errors';

type FileStatus = 'idle' | 'waiting' | 'uploading' | 'queued' | 'ok' | 'fail';

interface FileResult {
  status: FileStatus;
  name?: string;
  error?: string;
}

const BATCH_UPLOAD_SIZE = 10;
const POLL_INTERVAL_MS = 2500;
const MAX_POLL_CYCLES = 180; // up to ~7.5 minutes

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const isTransientError = (error: unknown) => {
  const status = getApiStatus(error);
  return status === undefined || status === 408 || status === 425 || status === 429 || status >= 500;
};

export default function BulkUpload() {
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<Record<string, FileResult>>({});
  const [uploading, setUploading] = useState(false);
  const [filterTab, setFilterTab] = useState<'all' | 'ok' | 'fail' | 'pending'>('all');
  const inputRef = useRef<HTMLInputElement>(null);
  const shouldStopRef = useRef(false);

  const handleSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selected = Array.from(e.target.files);
      setFiles(selected);
      const initialResults: Record<string, FileResult> = {};
      selected.forEach((f) => {
        initialResults[f.name] = { status: 'idle' };
      });
      setResults(initialResults);
    }
  };

  const uploadBatch = async (batchFiles: File[]): Promise<Array<{ task_id: string; filename: string }>> => {
    const queuedTasks: Array<{ task_id: string; filename: string }> = [];

    // Mark current batch as uploading
    setResults((prev) => {
      const updated = { ...prev };
      batchFiles.forEach((f) => {
        updated[f.name] = { status: 'uploading' };
      });
      return updated;
    });

    let uploaded = false;
    for (let retry = 0; retry < 3; retry += 1) {
      if (shouldStopRef.current) break;
      try {
        const formData = new FormData();
        batchFiles.forEach((file) => formData.append('files', file, file.name));

        const { data } = await api.post('/candidates/async/bulk', formData, {
          params: { use_llm: false },
          timeout: 120_000,
        });

        for (const task of (data.tasks || []) as Array<{
          task_id?: string;
          filename: string;
          status: string;
          error?: string;
        }>) {
          if (task.status === 'queued' && task.task_id) {
            queuedTasks.push({ task_id: task.task_id, filename: task.filename });
            setResults((prev) => ({
              ...prev,
              [task.filename]: { status: 'queued' },
            }));
          } else {
            setResults((prev) => ({
              ...prev,
              [task.filename]: { status: 'fail', error: task.error || 'Upload failed' },
            }));
          }
        }
        uploaded = true;
        break;
      } catch (err: unknown) {
        if (isTransientError(err) && retry < 2) {
          await sleep(2000 * (retry + 1));
          continue;
        }
        const message = getApiErrorMessage(err, 'Batch upload failed');
        setResults((prev) => {
          const updated = { ...prev };
          batchFiles.forEach((f) => {
            updated[f.name] = { status: 'fail', error: message };
          });
          return updated;
        });
        break;
      }
    }

    if (!uploaded && queuedTasks.length === 0) {
      setResults((prev) => {
        const updated = { ...prev };
        batchFiles.forEach((f) => {
          if (updated[f.name]?.status === 'uploading') {
            updated[f.name] = { status: 'fail', error: 'Upload failed' };
          }
        });
        return updated;
      });
    }

    return queuedTasks;
  };

  const pollTasksUntilDone = async (tasks: Array<{ task_id: string; filename: string }>) => {
    const pending = new Map(tasks.map((t) => [t.task_id, t.filename]));

    for (let cycle = 0; cycle < MAX_POLL_CYCLES; cycle += 1) {
      if (shouldStopRef.current || pending.size === 0) break;

      const taskIds = Array.from(pending.keys());

      // Query in chunks of max 40 IDs per status request
      for (let i = 0; i < taskIds.length; i += 40) {
        if (shouldStopRef.current) break;
        const chunk = taskIds.slice(i, i + 40);

        try {
          // Attempt POST status request first
          let statusMap: Record<string, { status: string; full_name?: string; error?: string }> = {};

          try {
            const res = await api.post<Record<string, { status: string; full_name?: string; error?: string }>>(
              '/candidates/async/bulk/status',
              { task_ids: chunk },
              { timeout: 25_000 }
            );
            statusMap = res.data;
          } catch {
            // Fallback to GET
            const res = await api.get<Record<string, { status: string; full_name?: string; error?: string }>>(
              '/candidates/async/bulk',
              {
                params: { task_ids: chunk },
                paramsSerializer: () => chunk.map((id) => `task_ids=${encodeURIComponent(id)}`).join('&'),
                timeout: 25_000,
              }
            );
            statusMap = res.data;
          }

          for (const [taskId, result] of Object.entries(statusMap)) {
            const filename = pending.get(taskId);
            if (!filename) continue;

            if (result.status === 'completed') {
              setResults((prev) => ({
                ...prev,
                [filename]: {
                  status: 'ok',
                  name: result.full_name || filename.replace(/\.[^/.]+$/, ''),
                },
              }));
              pending.delete(taskId);
            } else if (result.status === 'failed') {
              setResults((prev) => ({
                ...prev,
                [filename]: {
                  status: 'fail',
                  error: result.error || 'CV parsing failed',
                },
              }));
              pending.delete(taskId);
            }
          }
        } catch {
          // Temporary polling failure: DO NOT mark tasks as failed!
          // Just continue to next polling cycle.
        }
      }

      if (pending.size > 0) {
        await sleep(POLL_INTERVAL_MS);
      }
    }

    // Timeout remaining pending tasks
    if (!shouldStopRef.current) {
      for (const filename of pending.values()) {
        setResults((prev) => ({
          ...prev,
          [filename]: { status: 'fail', error: 'Processing timed out' },
        }));
      }
    }
  };

  const processQueue = async (filesToUpload: File[]) => {
    if (filesToUpload.length === 0) return;
    setUploading(true);
    shouldStopRef.current = false;

    // Set initial status to waiting
    setResults((prev) => {
      const updated = { ...prev };
      filesToUpload.forEach((f) => {
        updated[f.name] = { status: 'waiting' };
      });
      return updated;
    });

    const allQueuedTasks: Array<{ task_id: string; filename: string }> = [];

    // Split files into batches of 10 to prevent timeout/large payload errors
    for (let i = 0; i < filesToUpload.length; i += BATCH_UPLOAD_SIZE) {
      if (shouldStopRef.current) break;
      const batch = filesToUpload.slice(i, i + BATCH_UPLOAD_SIZE);
      const queued = await uploadBatch(batch);
      allQueuedTasks.push(...queued);
    }

    if (allQueuedTasks.length > 0 && !shouldStopRef.current) {
      await pollTasksUntilDone(allQueuedTasks);
    }

    setUploading(false);
  };

  const handleUploadAll = async () => {
    await processQueue(files);
  };

  const handleRetryFailed = async () => {
    const failedFiles = files.filter((f) => results[f.name]?.status === 'fail');
    if (failedFiles.length > 0) {
      await processQueue(failedFiles);
    }
  };

  const handleStop = () => {
    shouldStopRef.current = true;
    setUploading(false);
    setResults((prev) => {
      const updated = { ...prev };
      Object.keys(updated).forEach((name) => {
        if (
          updated[name].status === 'waiting' ||
          updated[name].status === 'uploading' ||
          updated[name].status === 'queued'
        ) {
          updated[name] = { status: 'idle', error: 'Cancelled by user' };
        }
      });
      return updated;
    });
  };

  const handleClear = () => {
    setFiles([]);
    setResults({});
    setFilterTab('all');
  };

  const doneCount = files.filter((f) => results[f.name]?.status === 'ok').length;
  const failedCount = files.filter((f) => results[f.name]?.status === 'fail').length;
  const inProgressCount = files.filter(
    (f) =>
      results[f.name]?.status === 'waiting' ||
      results[f.name]?.status === 'uploading' ||
      results[f.name]?.status === 'queued'
  ).length;

  const progressPercent =
    files.length > 0
      ? Math.round(((doneCount + failedCount) / files.length) * 100)
      : 0;

  const filteredFiles = files.filter((f) => {
    const status = results[f.name]?.status || 'idle';
    if (filterTab === 'ok') return status === 'ok';
    if (filterTab === 'fail') return status === 'fail';
    if (filterTab === 'pending')
      return status === 'waiting' || status === 'uploading' || status === 'queued';
    return true;
  });

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Bulk Upload CVs</h1>
        <p className="text-sm text-gray-500 mt-1">
          Upload and index large batches of candidate CVs automatically with reliable background processing.
        </p>
      </div>

      <div className="bg-white rounded-xl shadow-sm border p-6 space-y-6">
        <div
          className="border-2 border-dashed border-blue-200 rounded-xl p-8 text-center hover:border-blue-500 hover:bg-blue-50/30 transition-all cursor-pointer"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
              const dropped = Array.from(e.dataTransfer.files);
              setFiles(dropped);
              const initialResults: Record<string, FileResult> = {};
              dropped.forEach((f) => {
                initialResults[f.name] = { status: 'idle' };
              });
              setResults(initialResults);
            }
          }}
        >
          <Upload className="w-12 h-12 mx-auto mb-3 text-blue-500" />
          <p className="text-base font-semibold text-gray-800 mb-1">
            Drag & drop CV files here, or click to select
          </p>
          <p className="text-xs text-gray-400 mb-4">
            PDF, DOCX, TXT — up to 15MB each (supports 100+ files)
          </p>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.txt"
            onClick={(e) => {
              (e.target as HTMLInputElement).value = '';
            }}
            onChange={handleSelect}
            className="hidden"
          />
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              inputRef.current?.click();
            }}
            className="px-5 py-2.5 bg-blue-600 text-white font-medium text-sm rounded-lg hover:bg-blue-700 shadow-sm transition-colors"
          >
            Select Files
          </button>
        </div>

        {files.length > 0 && (
          <div className="space-y-4 pt-2 border-t">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-sm font-medium text-gray-900">
                  {files.length} CV{files.length > 1 ? 's' : ''} selected
                </p>
                <p className="text-xs text-gray-500">
                  Batch pipeline: chunked uploads with resilient status polling
                </p>
              </div>

              <div className="flex items-center gap-2">
                {!uploading ? (
                  <>
                    <button
                      onClick={handleUploadAll}
                      className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white text-sm font-medium rounded-lg hover:bg-emerald-700 shadow-sm transition-colors"
                    >
                      <Sparkles className="w-4 h-4" />
                      Upload All ({files.length})
                    </button>
                    {failedCount > 0 && (
                      <button
                        onClick={handleRetryFailed}
                        className="inline-flex items-center gap-2 px-3 py-2 bg-amber-500 text-white text-sm font-medium rounded-lg hover:bg-amber-600 transition-colors"
                      >
                        <RefreshCw className="w-4 h-4" />
                        Retry Failed ({failedCount})
                      </button>
                    )}
                    <button
                      onClick={handleClear}
                      className="px-3 py-2 border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      Clear
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      disabled
                      className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg opacity-80 cursor-not-allowed"
                    >
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Processing ({inProgressCount} in queue)...
                    </button>
                    <button
                      onClick={handleStop}
                      className="inline-flex items-center gap-1.5 px-3 py-2 bg-rose-600 text-white text-sm font-medium rounded-lg hover:bg-rose-700 transition-colors"
                    >
                      <Square className="w-4 h-4" />
                      Stop
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Progress bar */}
            {(uploading || doneCount > 0 || failedCount > 0) && (
              <div className="space-y-2 bg-gray-50 p-4 rounded-xl border">
                <div className="flex items-center justify-between text-xs font-semibold text-gray-700">
                  <span>
                    Progress: {doneCount + failedCount} of {files.length} processed ({progressPercent}%)
                  </span>
                  <div className="flex items-center gap-3">
                    <span className="text-emerald-700 flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> {doneCount} success
                    </span>
                    {failedCount > 0 && (
                      <span className="text-rose-700 flex items-center gap-1">
                        <XCircle className="w-3.5 h-3.5 text-rose-600" /> {failedCount} failed
                      </span>
                    )}
                    {inProgressCount > 0 && (
                      <span className="text-blue-700 flex items-center gap-1">
                        <Loader2 className="w-3.5 h-3.5 text-blue-600 animate-spin" /> {inProgressCount} in queue
                      </span>
                    )}
                  </div>
                </div>

                <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
                  <div
                    className="bg-emerald-600 h-2.5 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {files.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border p-6 space-y-4">
          <div className="flex items-center justify-between border-b pb-3">
            <div className="flex gap-2">
              <button
                onClick={() => setFilterTab('all')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
                  filterTab === 'all'
                    ? 'bg-blue-100 text-blue-800'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                All ({files.length})
              </button>
              <button
                onClick={() => setFilterTab('ok')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
                  filterTab === 'ok'
                    ? 'bg-emerald-100 text-emerald-800'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Success ({doneCount})
              </button>
              {failedCount > 0 && (
                <button
                  onClick={() => setFilterTab('fail')}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
                    filterTab === 'fail'
                      ? 'bg-rose-100 text-rose-800'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  Failed ({failedCount})
                </button>
              )}
              {inProgressCount > 0 && (
                <button
                  onClick={() => setFilterTab('pending')}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
                    filterTab === 'pending'
                      ? 'bg-amber-100 text-amber-800'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  In Progress / Queued ({inProgressCount})
                </button>
              )}
            </div>

            <span className="text-xs text-gray-400">
              Showing {filteredFiles.length} file{filteredFiles.length > 1 ? 's' : ''}
            </span>
          </div>

          <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
            {filteredFiles.map((file) => {
              const res = results[file.name] || { status: 'idle' };

              return (
                <div
                  key={file.name}
                  className="bg-white rounded-lg border p-3 flex items-center justify-between gap-3 hover:bg-gray-50/50 transition-colors"
                >
                  <div className="flex items-center gap-3 min-w-0 flex-1">
                    {res.status === 'idle' && (
                      <FileText className="w-5 h-5 text-gray-400 shrink-0" />
                    )}
                    {res.status === 'waiting' && (
                      <Clock className="w-5 h-5 text-amber-500 shrink-0" />
                    )}
                    {res.status === 'uploading' && (
                      <Loader2 className="w-5 h-5 text-blue-500 animate-spin shrink-0" />
                    )}
                    {res.status === 'queued' && (
                      <Loader2 className="w-5 h-5 text-indigo-500 animate-spin shrink-0" />
                    )}
                    {res.status === 'ok' && (
                      <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0" />
                    )}
                    {res.status === 'fail' && (
                      <XCircle className="w-5 h-5 text-rose-500 shrink-0" />
                    )}

                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {file.name}
                      </p>
                      {res.status === 'ok' && (
                        <p className="text-xs text-emerald-600 font-medium truncate">
                          ✓ Extracted: {res.name}
                        </p>
                      )}
                      {res.status === 'waiting' && (
                        <p className="text-xs text-amber-600">Waiting in queue...</p>
                      )}
                      {res.status === 'uploading' && (
                        <p className="text-xs text-blue-600">Uploading to server...</p>
                      )}
                      {res.status === 'queued' && (
                        <p className="text-xs text-indigo-600">
                          Extracting skills and indexing profile in background...
                        </p>
                      )}
                      {res.status === 'fail' && (
                        <p className="text-xs text-rose-600 flex items-center gap-1 truncate">
                          <AlertTriangle className="w-3 h-3 shrink-0" />
                          {res.error || 'Failed'}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="shrink-0 text-xs">
                    {res.status === 'ok' && (
                      <span className="px-2.5 py-1 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-md font-medium">
                        Done
                      </span>
                    )}
                    {res.status === 'fail' && (
                      <span className="px-2.5 py-1 bg-rose-50 text-rose-700 border border-rose-200 rounded-md font-medium">
                        Failed
                      </span>
                    )}
                    {res.status === 'queued' && (
                      <span className="px-2.5 py-1 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-md font-medium">
                        Processing
                      </span>
                    )}
                    {res.status === 'uploading' && (
                      <span className="px-2.5 py-1 bg-blue-50 text-blue-700 border border-blue-200 rounded-md font-medium">
                        Uploading
                      </span>
                    )}
                    {res.status === 'waiting' && (
                      <span className="px-2.5 py-1 bg-amber-50 text-amber-700 border border-amber-200 rounded-md font-medium">
                        Queued
                      </span>
                    )}
                    {res.status === 'idle' && (
                      <span className="px-2.5 py-1 bg-gray-50 text-gray-600 border border-gray-200 rounded-md font-medium">
                        Ready
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
