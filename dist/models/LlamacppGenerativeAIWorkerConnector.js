import { EventEmitter } from 'events';
/**
 * The Llamacpp machine learning connector.
 */
export class LlamacppGenerativeAIWorkerConnector {
    options;
    /**
     * The vector database connector.
     * @ignore
     */
    vectorDatabase;
    /**
     * The instructions.
     * @ignore
     */
    instructions = 'Keep the answer short and concise.';
    /**
     * The Llama engine.
     * @ignore
     */
    engine;
    /**
     * The model to use for language modeling.
     * @ignore
     */
    llmModel;
    /**
     * The model to use for sentence similarity.
     * @ignore
     */
    similarityModel;
    /**
     * The sentences.
     * @ignore
     */
    sentences = [];
    /**
     * The documents.
     * @ignore
     */
    documents = [];
    /**
     * The embedding context.
     * @ignore
     */
    embeddingContext;
    /**
     * The chat context.
     * @ignore
     */
    chatContext;
    /**
     * The chat session.
     * @ignore
     */
    chatSession;
    /**
     * The constructor.
     * @param vectorDatabaseConnector The vector database connector constructor.
     */
    constructor(vectorDatabaseConnector, options) {
        this.options = options;
        this.vectorDatabase = new vectorDatabaseConnector();
    }
    /**
     * Initialize the machine learning model.
     * @param llmModel The path to the LLM model.
     * @param similarityModel The path to the similarity model.
     */
    async initialize(llmModel, similarityModel) {
        if (!llmModel && !this.options?.llmPath) {
            throw new Error('LLM model path not provided');
        }
        if (!similarityModel && !this.options?.similarityPath) {
            throw new Error('Similarity model path not provided');
        }
        const { getLlama } = await import('node-llama-cpp');
        this.engine = await getLlama();
        this.llmModel = await this.engine.loadModel({
            modelPath: llmModel ?? this.options?.llmPath ?? 'llm.gguf',
        });
        this.similarityModel = await this.engine.loadModel({
            modelPath: similarityModel ?? this.options?.similarityPath ?? 'similarity.gguf',
        });
    }
    /**
     * Add content to the machine learning model.
     * @param name The name of the content.
     * @param content The content to add.
     * @returns A promise that resolves when the content has been added.
     */
    async addContent(name, content) {
        if (!this.similarityModel) {
            throw new Error('Model not initialized');
        }
        this.embeddingContext = await this.similarityModel.createEmbeddingContext();
        const sentences = this.splitIntoSentences(content);
        this.documents.push({
            name,
            startIndex: this.sentences.length,
            length: sentences.length,
        });
        this.sentences.push(...sentences);
        const embeddings = await Promise.all(sentences.map(sentence => this.getVector(sentence)));
        this.vectorDatabase.insert(embeddings);
        await this.embeddingContext.dispose();
        this.embeddingContext = undefined;
    }
    /**
     * Remove content from the machine learning model.
     * @param name The name of the content.
     */
    removeContent(name) {
        const document = this.documents.find(document => document.name === name);
        if (!document)
            return;
        this.vectorDatabase.remove(Array.from({ length: document.length }, (_, i) => document.startIndex + i));
        this.sentences.splice(document.startIndex, document.length);
        this.documents = this.documents.filter(doc => doc !== document);
    }
    /**
     * Prompt the machine learning model.
     * @param prompt The prompt to use.
     * @returns An async generator that yields the responses.
     */
    async *processJob(job) {
        if (!this.llmModel) {
            throw new Error('Model not initialized');
        }
        if (job.parameters.jobType !== 0) {
            return;
        }
        const { LlamaChatSession } = await import('node-llama-cpp');
        const prompt = job.parameters.prompt;
        const context = await this.getContext(prompt);
        if (!this.chatContext) {
            this.chatContext = await this.llmModel.createContext();
        }
        if (!this.chatSession) {
            this.chatSession = new LlamaChatSession({
                contextSequence: this.chatContext.getSequence(),
            });
        }
        const tokenEmitter = new EventEmitter();
        this.chatSession.prompt(`Instructions: ${this.instructions}, Context: ${context}, Prompt: ${prompt}`, {
            maxTokens: 1024,
            temperature: 0.5,
            onToken: async (token) => {
                tokenEmitter.emit('token', token);
            }
        });
        while (true) {
            const token = await new Promise((resolve) => tokenEmitter.once('token', resolve));
            if (!token)
                break;
            yield {
                id: job.id,
                status: 'Completed',
                result: {
                    jobType: 0,
                    output: this.llmModel.detokenize(token),
                },
            };
        }
    }
    /**
     * Get the context for a prompt.
     * @param prompt The prompt.
     * @returns A promise that resolves to the context.
     * @ignore
     */
    async getContext(prompt) {
        if (!this.similarityModel) {
            throw new Error('Model not initialized');
        }
        this.embeddingContext = await this.similarityModel.createEmbeddingContext();
        const embedding = await this.getVector(prompt);
        const context = this.vectorDatabase.search(embedding, 5).map(index => this.sentences.slice(index - 2, index + 3).join(' '));
        await this.embeddingContext.dispose();
        return context.join(' ');
    }
    /**
     * Get the vector for some content.
     * @param content The content.
     * @returns A promise that resolves to the embedding.
     * @ignore
     */
    async getVector(content) {
        if (!this.llmModel) {
            throw new Error('Model not initialized');
        }
        if (!this.embeddingContext) {
            throw new Error('Embedding context not initialized');
        }
        return (await this.embeddingContext.getEmbeddingFor(content)).vector;
    }
    /**
     * Split text into sentences.
     * @param text The text to split.
     * @returns The sentences.
     * @ignore
     */
    splitIntoSentences(text) {
        return text.match(/[^\.!\?]+[\.!\?]+/g)?.filter(sentence => sentence.trim().length > 0) || [];
    }
}
