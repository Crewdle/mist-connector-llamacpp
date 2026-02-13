import type { GenerativeAIEngineType, IGenerativeAIModel, IGenerativeAIWorkerConnector, IGenerativeAIWorkerOptions, GenerativeAIWorkerConnectorParameters, GenerativeAIWorkerConnectorResult, IGenerativeAIPromptWorkerConnectorParameters, IGenerativeAIWorkerConnectorPromptResult } from '@crewdle/web-sdk-types';
import { ILlamacppGenerativeAIWorkerOptions } from './LlamacppGenerativeAIWorkerOptions';
import { ILlamacppGenerativeAIWorkerModel } from './LlamacppGenerativeAIWorkerModel';
/**
 * The Llamacpp machine learning connector.
 */
export declare class LlamacppGenerativeAIWorkerConnector implements IGenerativeAIWorkerConnector {
    private apiKey?;
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
     * Microsoft Graph OAuth tokens.
     * @ignore
     */
    private static microsoftGraphTokens?;
    /**
     * Callback to refresh Microsoft Graph access tokens.
     * @ignore
     */
    private static refreshTokensCallback?;
    /**
     * Callback to search the web.
     * @ignore
     */
    private static browserCallback?;
    /**
     * The base folder.
     * @ignore
     */
    private baseFolder?;
    /**
     * The constructor.
     * @param options The options.
     */
    constructor(apiKey?: string | undefined, options?: ILlamacppGenerativeAIWorkerOptions | undefined);
    static getModels(): Map<string, ILlamacppGenerativeAIWorkerModel>;
    static downloadModel(id: string, baseFolder: string): Promise<void>;
    static loadModel(id: string): Promise<void>;
    /**
     * Get the VRAM state.
     * @returns The total and available VRAM.
     */
    static getVramState(): Promise<{
        total: number;
        available: number;
    }>;
    /**
     * Set the Microsoft Graph OAuth tokens.
     * @param tokens The OAuth tokens for Microsoft Graph.
     */
    static setMicrosoftGraphTokens(tokens: {
        accessToken: string;
        refreshToken?: string;
        tokenType?: string;
        expiresAt?: number;
        scope?: string;
    } | undefined): void;
    /**
     * Set the callback to refresh Microsoft Graph access tokens.
     * @param callback The callback function to refresh access tokens.
     */
    static setRefreshTokensCallback(callback: (refreshToken: string) => Promise<{
        accessToken: string;
        refreshToken?: string;
        tokenType?: string;
        expiresAt?: number;
        scope?: string;
    }>): void;
    /**
     * Set the callback to browse the web.
     * @param callback The callback function to browse the web.
     */
    static setBrowserCallback(callback: (query: string) => Promise<string>): void;
    /**
     * Get the Microsoft Graph OAuth tokens.
     * Automatically refreshes tokens if they are expired or about to expire (within 5 minutes).
     * @returns A promise that resolves with the OAuth tokens if set, undefined otherwise.
     */
    private static getMicrosoftGraphTokens;
    /**
     * Refresh the Microsoft Graph access tokens.
     * @returns A promise that resolves with the refreshed access tokens.
     * @throws Error if no refresh callback is set or no refresh token is available.
     */
    private static refreshAccessTokens;
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
     * @param modelId The model ID to get context size from
     * @returns A promise that resolves to the functions object
     */
    private getFunctions;
    private markdownSearch;
    private markdownContent;
}
