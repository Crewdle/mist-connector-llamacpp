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
    maxContents = 5;
    /**
     * The number of sentences to include in one content.
     * @ignore
     */
    maxSentences = 6;
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
        if (options?.maxSentences) {
            this.maxSentences = options.maxSentences;
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
        const sentences = this.splitIntoSentences(content);
        this.documents.push({
            name,
            startIndex: this.sentences.length,
            length: sentences.length,
        });
        this.sentences.push(...sentences);
        const embeddings = [];
        for (let i = 0; i < sentences.length; i++) {
            try {
                const embedding = await this.getVector(sentences[i]);
                embeddings.push(embedding);
            }
            catch (e) { }
        }
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
            maxTokens: this.maxTokens,
            temperature: this.temperature,
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
            maxTokens: this.maxTokens,
            temperature: this.temperature,
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
        const { useRAG, prompt, instructions, context } = job.parameters;
        let finalPrompt = `${instructions ?? this.instructions}\n\n`;
        if (useRAG === undefined || useRAG === true) {
            let ragContext = '';
            ragContext = await this.getContext(prompt);
            if (ragContext !== '') {
                finalPrompt += `Relevant Information:\n\n${ragContext}\n\n`;
            }
        }
        finalPrompt += `Conversation:\n`;
        if (context) {
            let conversation = '';
            let conversationLength = this.llmModel.tokenize(finalPrompt).length + this.llmModel.tokenize(`Human: ${prompt}\nAI:`).length;
            for (let i = context.length - 1; i >= 0; i--) {
                const { source, message } = context[i];
                if (conversationLength + this.llmModel.tokenize(message).length > this.llmModel.trainContextSize) {
                    break;
                }
                conversation = `${source === 'ai' ? 'AI' : 'Human'}: ${message}\n${conversation}`;
                conversationLength += this.llmModel.tokenize(message).length;
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
    async getContext(prompt) {
        if (!this.similarityModel) {
            throw new Error('Model not initialized');
        }
        this.embeddingContext = await this.similarityModel.createEmbeddingContext();
        const embedding = await this.getVector(prompt);
        const context = this.vectorDatabase.search(embedding, this.maxContents).map(index => {
            let context = '';
            const document = this.documents.find(document => document.startIndex <= index && index < document.startIndex + document.length);
            if (document) {
                context += `From ${document.name}:\n`;
            }
            context += this.sentences.slice(index - (this.maxSentences / 2 - 1), index + (this.maxSentences / 2)).join(' ');
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
