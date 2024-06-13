import type { IGenerativeAIWorkerConnector, IJobParametersAI, IJobRequest, IJobResultAI, JobResponse, VectorDatabaseConnectorConstructor } from '@crewdle/web-sdk-types';
import { ILlamacppGenerativeAIWorkerOptions } from './LlamacppGenerativeAIWorkerOptions';
/**
 * The Llamacpp machine learning connector.
 */
export declare class LlamacppGenerativeAIWorkerConnector implements IGenerativeAIWorkerConnector {
    private options?;
    /**
     * The vector database connector.
     * @ignore
     */
    private vectorDatabase;
    /**
     * The instructions.
     * @ignore
     */
    private instructions;
    /**
     * The maximum number of tokens to generate.
     * @ignore
     */
    private maxTokens;
    /**
     * The temperature.
     * @ignore
     */
    private temperature;
    /**
     * The number of contents to include in the context.
     * @ignore
     */
    private maxContents;
    /**
     * The number of sentences to include in one content.
     * @ignore
     */
    private maxSentences;
    /**
     * The Llama engine.
     * @ignore
     */
    private engine?;
    /**
     * The model to use for language modeling.
     * @ignore
     */
    private llmModel?;
    /**
     * The model to use for sentence similarity.
     * @ignore
     */
    private similarityModel?;
    /**
     * The sentences.
     * @ignore
     */
    private sentences;
    /**
     * The documents.
     * @ignore
     */
    private documents;
    /**
     * The embedding context.
     * @ignore
     */
    private embeddingContext?;
    /**
     * The chat context.
     * @ignore
     */
    private chatContext?;
    /**
     * The chat session.
     * @ignore
     */
    private chatSession?;
    /**
     * The constructor.
     * @param vectorDatabaseConnector The vector database connector constructor.
     */
    constructor(vectorDatabaseConnector: VectorDatabaseConnectorConstructor, options?: ILlamacppGenerativeAIWorkerOptions | undefined);
    /**
     * Initialize the machine learning model.
     * @param llmModel The path to the LLM model.
     * @param similarityModel The path to the similarity model.
     */
    initialize(llmModel?: string, similarityModel?: string): Promise<void>;
    /**
     * Add content to the machine learning model.
     * @param name The name of the content.
     * @param content The content to add.
     * @returns A promise that resolves when the content has been added.
     */
    addContent(name: string, content: string): Promise<void>;
    /**
     * Remove content from the machine learning model.
     * @param name The name of the content.
     */
    removeContent(name: string): void;
    /**
     * Process a job.
     * @param job The job to process.
     * @returns A promise that resolves with the job result.
     */
    processJob(job: IJobRequest<IJobParametersAI>): Promise<JobResponse<IJobResultAI>>;
    /**
     * Stream a job.
     * @param job The job to stream.
     * @returns An async generator that yields the responses.
     */
    processJobStream(job: IJobRequest<IJobParametersAI>): AsyncGenerator<JobResponse<IJobResultAI>>;
    private getPrompt;
    /**
     * Get the context for a prompt.
     * @param prompt The prompt.
     * @returns A promise that resolves to the context.
     * @ignore
     */
    private getContext;
    /**
     * Get the vector for some content.
     * @param content The content.
     * @returns A promise that resolves to the embedding.
     * @ignore
     */
    private getVector;
    /**
     * Clean the text.
     * @param text The text to clean.
     * @returns The cleaned text.
     * @ignore
     */
    private cleanText;
    /**
     * Normalize a vector.
     * @param vector The vector to normalize.
     * @returns The normalized vector.
     * @ignore
     */
    private normalizeVector;
}
