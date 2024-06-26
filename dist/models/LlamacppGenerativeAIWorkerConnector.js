import { EventEmitter } from 'events';
const CHUNK_MAX_LENGTH = 512;
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
    instructions = `The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it will use concepts stored in memory that have very similar weights.`;
    /**
     * The maximum number of tokens to generate.
     * @ignore
     */
    maxTokens = 1024;
    /**
     * The temperature.
     * @ignore
     */
    temperature = 1;
    /**
     * The number of contents to include in the context.
     * @ignore
     */
    maxContents = 10;
    /**
     * The number of chunks to include in one content.
     * @ignore
     */
    maxChunks = 2;
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
     * The chunks of documents.
     * @ignore
     */
    chunks = [];
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
     * @param options The options.
     */
    constructor(vectorDatabaseConnector, options) {
        this.options = options;
        this.vectorDatabase = new vectorDatabaseConnector();
        if (options?.instructions) {
            this.instructions = options.instructions;
        }
        if (options?.maxTokens) {
            this.maxTokens = options.maxTokens;
        }
        if (options?.temperature) {
            this.temperature = options.temperature;
        }
        if (options?.maxContents) {
            this.maxContents = options.maxContents;
        }
        if (options?.maxChunks) {
            this.maxChunks = options.maxChunks;
        }
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
        const document = this.documents.find(document => document.name === name);
        if (document) {
            this.removeContent(name);
        }
        this.embeddingContext = await this.similarityModel.createEmbeddingContext();
        const embeddings = [];
        try {
            for (let j = 0; j < content.length; j += CHUNK_MAX_LENGTH) {
                const embedding = await this.getVector(content.slice(j, j + CHUNK_MAX_LENGTH));
                this.chunks.push(content.slice(j, j + CHUNK_MAX_LENGTH));
                embeddings.push(embedding);
            }
        }
        catch (e) {
            console.error(e);
        }
        if (embeddings.length > 0) {
            this.documents.push({
                name,
                startIndex: this.chunks.length,
                length: embeddings.length,
            });
            this.vectorDatabase.insert(embeddings);
        }
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
        this.chunks.splice(document.startIndex, document.length);
        this.documents = this.documents.filter(doc => doc !== document);
    }
    /**
     * Process a job.
     * @param job The job to process.
     * @returns A promise that resolves with the job result.
     */
    async processJob(job) {
        if (!this.llmModel) {
            throw new Error('Model not initialized');
        }
        if (!this.chatContext) {
            this.chatContext = await this.llmModel.createContext();
        }
        if (!this.chatSession) {
            const { LlamaChatSession } = await import('node-llama-cpp');
            this.chatSession = new LlamaChatSession({
                contextSequence: this.chatContext.getSequence(),
            });
        }
        const prompt = await this.getPrompt(job);
        const output = await this.chatSession.prompt(prompt, {
            maxTokens: job.parameters.maxTokens ?? this.maxTokens,
            temperature: job.parameters.temperature ?? this.temperature,
        });
        const inputTokens = this.llmModel.tokenize(prompt).length;
        const outputTokens = this.llmModel.tokenize(output).length;
        this.chatSession.dispose();
        this.chatSession = undefined;
        this.chatContext.dispose();
        this.chatContext = undefined;
        return {
            id: job.id,
            status: 'completed',
            result: {
                output,
                inputTokens,
                outputTokens,
                contentSize: this.chunks.length,
            },
        };
    }
    /**
     * Stream a job.
     * @param job The job to stream.
     * @returns An async generator that yields the responses.
     */
    async *processJobStream(job) {
        if (!this.llmModel) {
            throw new Error('Model not initialized');
        }
        if (!this.chatContext) {
            this.chatContext = await this.llmModel.createContext();
        }
        if (!this.chatSession) {
            const { LlamaChatSession } = await import('node-llama-cpp');
            this.chatSession = new LlamaChatSession({
                contextSequence: this.chatContext.getSequence(),
            });
        }
        const tokenEmitter = new EventEmitter();
        const prompt = await this.getPrompt(job);
        const inputTokens = this.llmModel.tokenize(prompt).length;
        let outputTokens = 0;
        this.chatSession.prompt(prompt, {
            maxTokens: job.parameters.maxTokens ?? this.maxTokens,
            temperature: job.parameters.temperature ?? this.temperature,
            onToken: async (token) => {
                tokenEmitter.emit('token', token);
            }
        }).then(() => {
            tokenEmitter.emit('token', undefined);
        }).catch(e => { });
        while (true) {
            const token = await new Promise((resolve) => tokenEmitter.once('token', resolve));
            if (!token || this.llmModel.detokenize(token).indexOf('<|end|>') !== -1) {
                break;
            }
            outputTokens += token.length;
            yield {
                id: job.id,
                status: 'partial',
                result: {
                    output: this.llmModel.detokenize(token),
                    inputTokens,
                    outputTokens,
                    contentSize: this.chunks.length,
                },
            };
        }
        this.chatSession.dispose();
        this.chatSession = undefined;
        this.chatContext.dispose();
        this.chatContext = undefined;
    }
    async getPrompt(job) {
        if (!this.llmModel) {
            throw new Error('Model not initialized');
        }
        const { useRAG, prompt, instructions, history } = job.parameters;
        let finalPrompt = `${instructions ?? this.instructions}\n\n`;
        finalPrompt += `Relevant Information:\n\n`;
        if (useRAG === undefined || useRAG === true) {
            let ragContext = '';
            ragContext = await this.getContext(prompt, job.parameters);
            if (ragContext !== '') {
                finalPrompt += `${ragContext}\n\n`;
            }
            else {
                finalPrompt += `No relevant information found.\n\n`;
            }
        }
        else {
            finalPrompt += `No relevant information used.\n\n`;
        }
        finalPrompt += `Conversation:\n`;
        if (history) {
            let conversation = '';
            let conversationLength = this.llmModel.tokenize(finalPrompt).length + this.llmModel.tokenize(`Human: ${prompt}\nAI:`).length;
            for (let i = history.length - 1; i >= 0; i--) {
                const { source, message } = history[i];
                if (conversationLength + this.llmModel.tokenize(message).length > this.llmModel.trainContextSize * 0.75) {
                    break;
                }
                conversation = `${source === 'ai' ? 'AI' : 'Human'}: ${message}\n${conversation}`;
                conversationLength += this.llmModel.tokenize(`${source === 'ai' ? 'AI' : 'Human'}: ${message}\n`).length;
            }
            finalPrompt += conversation;
        }
        finalPrompt += `Human: ${prompt}\nAI:`;
        return finalPrompt;
    }
    /**
     * Get the context for a prompt.
     * @param prompt The prompt.
     * @returns A promise that resolves to the context.
     * @ignore
     */
    async getContext(prompt, options) {
        if (!this.similarityModel) {
            throw new Error('Model not initialized');
        }
        this.embeddingContext = await this.similarityModel.createEmbeddingContext();
        const embedding = await this.getVector(prompt);
        const maxChunks = options?.maxChunks ?? this.maxChunks;
        const context = this.vectorDatabase.search(embedding, options?.maxContents ?? this.maxContents, options?.minRelevance, options?.ragStartingIndex).map(index => {
            let context = '';
            const document = this.documents.find(document => document.startIndex <= index && index < document.startIndex + document.length);
            if (document) {
                context += `From ${document.name}:\n`;
            }
            context += this.chunks.slice(index - (maxChunks / 2 - 1), index + (maxChunks / 2)).join(' ');
            return context;
        });
        await this.embeddingContext.dispose();
        return context.join("\n");
    }
    /**
     * Get the vector for some content.
     * @param content The content.
     * @returns A promise that resolves to the embedding.
     * @ignore
     */
    async getVector(content) {
        if (!this.embeddingContext) {
            throw new Error('Embedding context not initialized');
        }
        const vector = (await this.embeddingContext.getEmbeddingFor(this.cleanText(content))).vector;
        return this.normalizeVector(vector);
    }
    /**
     * Clean the text.
     * @param text The text to clean.
     * @returns The cleaned text.
     * @ignore
     */
    cleanText(text) {
        return text.trim().toLowerCase().replace(/[^a-z0-9\s]/g, '');
    }
    /**
     * Normalize a vector.
     * @param vector The vector to normalize.
     * @returns The normalized vector.
     * @ignore
     */
    normalizeVector(vector) {
        const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
        return vector.map((value) => value / norm);
    }
}
