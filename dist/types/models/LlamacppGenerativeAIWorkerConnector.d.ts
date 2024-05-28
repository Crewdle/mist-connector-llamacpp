import { IJob, IJobResult, IGenerativeAIWorkerConnector, VectorDatabaseConnectorConstructor } from '@crewdle/web-sdk-types';
/**
 * The Llamacpp machine learning connector.
 * @category Connector
 */
export declare class LlamacppGenerativeAIWorkerConnector implements IGenerativeAIWorkerConnector {
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
     * @param vectorDatabaseConstructor The vector database connector constructor.
     */
    constructor(vectorDatabaseConstructor: VectorDatabaseConnectorConstructor);
    /**
     * Initialize the machine learning model.
     * @param llmModel The path to the LLM model.
     * @param similarityModel The path to the similarity model.
     */
    initialize(llmModel: string, similarityModel: string): Promise<void>;
    /**
     * Add content to the machine learning model.
     * @param content The content to add.
     * @returns A promise that resolves when the content has been added.
     */
    addContent(content: string): Promise<void>;
    /**
     * Prompt the machine learning model.
     * @param prompt The prompt to use.
     * @returns An async generator that yields the responses.
     */
    processJob(job: IJob): AsyncGenerator<IJobResult>;
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
     * Split text into sentences.
     * @param text The text to split.
     * @returns The sentences.
     * @ignore
     */
    private splitIntoSentences;
}
