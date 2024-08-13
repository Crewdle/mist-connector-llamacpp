import type { IGenerativeAIWorkerConnector, IGenerativeAIWorkerOptions, IJobParametersAI, IJobResultAI } from '@crewdle/web-sdk-types';
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
     * The context.
     * @ignore
     */
    private static context?;
    /**
     * The embedding context.
     * @ignore
     */
    private static embeddingContext?;
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
     * Set a model.
     * @param id The model ID.
     * @param model The model.
     * @ignore
     */
    private static setModel;
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
    initialize(workflowId: string, models: Map<string, string>): Promise<void>;
    /**
     * Close the machine learning model.
     * @returns A promise that resolves when the model has been closed.
     */
    close(): Promise<void>;
    /**
     * Process a job.
     * @param parameters The job parameters.
     * @returns A promise that resolves with the job result.
     */
    processJob(parameters: IJobParametersAI, options: IGenerativeAIWorkerOptions): Promise<IJobResultAI>;
    /**
     * Stream a job.
     * @param parameters The job parameters.
     * @returns An async generator that yields the responses.
     */
    processJobStream(parameters: IJobParametersAI, options: IGenerativeAIWorkerOptions): AsyncGenerator<IJobResultAI>;
    private getPrompt;
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
}
