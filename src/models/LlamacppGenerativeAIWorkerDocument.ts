/**
 * The interface for a LlamacppGenerativeAIWorker document.
 */
export interface ILlamacppGenerativeAIWorkerDocument {
  /**
   * The name of the document.
   */
  name: string;

  /**
   * The starting index of the document.
   */
  startIndex: number;

  /**
   * The length of the document.
   */
  length: number;
}
