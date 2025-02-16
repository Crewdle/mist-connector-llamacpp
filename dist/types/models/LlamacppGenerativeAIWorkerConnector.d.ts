import type { GenerativeAIEngineType, IGenerativeAIModel, IGenerativeAIWorkerConnector, IGenerativeAIWorkerOptions, GenerativeAIWorkerConnectorParameters, GenerativeAIWorkerConnectorResult, IGenerativeAIPromptWorkerConnectorParameters, IGenerativeAIWorkerConnectorPromptResult } from '@crewdle/web-sdk-types';
import { ILlamacppGenerativeAIWorkerOptions } from './LlamacppGenerativeAIWorkerOptions';
/**
 * The Llamacpp machine learning connector.
 */
export declare class LlamacppGenerativeAIWorkerConnector implements IGenerativeAIWorkerConnector {
    private options?;
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
     * The workflow ID.
     * @ignore
     */
    private workflowId?;
    /**
     * The Llama engine.
     * @ignore
     */
    private static engine?;
    /**
     * The models.
     * @ignore
     */
    private static models;
    /**
     * The embedding context.
     * @ignore
     */
    private static embeddingContext?;
    /**
     * The base folder.
     * @ignore
     */
    private baseFolder?;
    /**
     * The constructor.
     * @param options The options.
     */
    constructor(options?: ILlamacppGenerativeAIWorkerOptions | undefined);
    /**
     * Get the VRAM state.
     * @returns The total and available VRAM.
     */
    static getVramState(): Promise<{
        total: number;
        available: number;
    }>;
    /**
     * Get the Llama engine.
     * @returns A promise that resolves with the Llama engine.
     * @ignore
     */
    private static getEngine;
    /**
     * Get a model.
     * @param id The model ID.
     * @returns The model.
     * @ignore
     */
    private static getModel;
    /**
     * Get a model context.
     * @param id The model ID.
     * @returns The model context.
     * @ignore
     */
    private static getContext;
    /**
     * Set a model.
     * @param id The model ID.
     * @param model The model.
     * @ignore
     */
    private static setModel;
    /**
     * Set a model context.
     * @param id The model ID.
     * @param context The model context.
     * @ignore
     */
    private static setContext;
    /**
     * Delete a model.
     * @param id The model ID.
     * @ignore
     */
    private static deleteModel;
    /**
     * Initialize the machine learning model.
     * @param workflowId The workflow ID.
     * @param models The models to initialize.
     */
    initialize(workflowId: string, models: Map<string, IGenerativeAIModel>): Promise<void>;
    /**
     * Close the machine learning model.
     * @returns A promise that resolves when the model has been closed.
     */
    close(): Promise<void>;
    getEngineType(): GenerativeAIEngineType;
    /**
     * Process a job.
     * @param parameters The job parameters.
     * @returns A promise that resolves with the job result.
     */
    processJob(parameters: GenerativeAIWorkerConnectorParameters, options: IGenerativeAIWorkerOptions): Promise<GenerativeAIWorkerConnectorResult>;
    /**
     * Stream a job.
     * @param parameters The job parameters.
     * @returns An async generator that yields the responses.
     */
    processJobStream(parameters: IGenerativeAIPromptWorkerConnectorParameters, options: IGenerativeAIWorkerOptions): AsyncGenerator<IGenerativeAIWorkerConnectorPromptResult>;
    private setupSession;
    /**
     * Get the vector for some content.
     * @param embeddingContext The embedding context.
     * @param content The content.
     * @returns A promise that resolves to the vector.
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
    /**
     * Get the functions object.
     * @param functions The functions to process
     * @returns A promise that resolves to the functions object
     */
    private getFunctions;
}
