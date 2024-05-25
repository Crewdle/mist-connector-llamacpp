import { IMachineLearningConnector, VectorDatabaseConnectorConstructor } from '@crewdle/web-sdk-types';
export declare class LlamacppMachineLearningConnector implements IMachineLearningConnector {
    private vectorDatabase;
    private instructions;
    private engine?;
    private llmModel?;
    private similarityModel?;
    private sentences;
    private embeddingContext?;
    private chatContext?;
    private chatSession?;
    constructor(vectorDatabaseConstructor: VectorDatabaseConnectorConstructor);
    initialize(llmModel: string, similarityModel: string): Promise<void>;
    addContent(content: string): Promise<void>;
    prompt(prompt: string): AsyncGenerator<string>;
    private getContext;
    private getEmbedding;
    private splitIntoSentences;
}
