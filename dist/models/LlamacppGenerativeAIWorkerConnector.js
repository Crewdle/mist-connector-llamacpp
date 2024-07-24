import { EventEmitter } from 'events';
/**
 * The Llamacpp machine learning connector.
 */
export class LlamacppGenerativeAIWorkerConnector {
    options;
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
     * The Llama engine.
     * @ignore
     */
    static engine;
    /**
     * The models.
     */
    models = new Map();
    /**
     * The constructor.
     * @param options The options.
     */
    constructor(options) {
        this.options = options;
        if (this.options?.instructions) {
            this.instructions = this.options.instructions;
        }
        if (this.options?.maxTokens) {
            this.maxTokens = this.options.maxTokens;
        }
        if (this.options?.temperature) {
            this.temperature = this.options.temperature;
        }
    }
    static async getEngine() {
        if (this.engine) {
            return this.engine;
        }
        const { getLlama } = await import('node-llama-cpp');
        this.engine = await getLlama();
        return this.engine;
    }
    /**
     * Initialize the machine learning model.
     * @param models The models to initialize.
     */
    async initialize(models) {
        const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
        for (const [modelName, modelPath] of models) {
            this.models.set(modelName, await engine.loadModel({ modelPath }));
        }
    }
    /**
     * Close the machine learning model.
     * @returns A promise that resolves when the model has been closed.
     */
    async close() {
        for (const model of this.models.values()) {
            await model.dispose();
        }
    }
    /**
     * Process a job.
     * @param parameters The job parameters.
     * @returns A promise that resolves with the job result.
     */
    async processJob(parameters, options) {
        const model = this.models.get(options.model.id);
        if (!model) {
            throw new Error('Model not initialized');
        }
        if (options.model.outputType === 'vector') {
            const context = await model.createEmbeddingContext();
            const vector = await this.getVector(context, parameters.prompt);
            await context.dispose();
            return {
                output: vector,
            };
        }
        const context = await model.createContext();
        const { LlamaChatSession } = await import('node-llama-cpp');
        const session = new LlamaChatSession({
            contextSequence: context.getSequence(),
        });
        const prompt = await this.getPrompt(parameters, model);
        const output = await session.prompt(prompt, {
            maxTokens: parameters.maxTokens ?? this.maxTokens,
            temperature: parameters.temperature ?? this.temperature,
        });
        const inputTokens = model.tokenize(prompt).length;
        const outputTokens = model.tokenize(output).length;
        session.dispose();
        await context.dispose();
        return {
            output,
            inputTokens,
            outputTokens,
        };
    }
    /**
     * Stream a job.
     * @param parameters The job parameters.
     * @returns An async generator that yields the responses.
     */
    async *processJobStream(parameters, options) {
        const model = this.models.get(options.model.id);
        if (!model) {
            throw new Error('Model not initialized');
        }
        if (options.model.outputType === 'vector') {
            throw new Error('Vector output type not supported for streaming');
        }
        const context = await model.createContext();
        const { LlamaChatSession } = await import('node-llama-cpp');
        const session = new LlamaChatSession({
            contextSequence: context.getSequence(),
        });
        const prompt = await this.getPrompt(parameters, model);
        const inputTokens = model.tokenize(prompt).length;
        let outputTokens = 0;
        const tokenEmitter = new EventEmitter();
        session.prompt(prompt, {
            maxTokens: parameters.maxTokens ?? this.maxTokens,
            temperature: parameters.temperature ?? this.temperature,
            onToken: async (token) => {
                tokenEmitter.emit('token', token);
            }
        }).then(() => {
            tokenEmitter.emit('token', undefined);
        }).catch(e => { });
        while (true) {
            const token = await new Promise((resolve) => tokenEmitter.once('token', resolve));
            if (token === undefined) {
                break;
            }
            const output = model.detokenize(token);
            if (output.indexOf('<|end|>') !== -1) {
                break;
            }
            outputTokens += token.length;
            yield {
                output,
                inputTokens,
                outputTokens,
            };
        }
        session.dispose();
        await context.dispose();
    }
    async getPrompt(parameters, model) {
        const { prompt, instructions, history } = parameters;
        let finalPrompt = `${instructions ?? this.instructions}\n\n`;
        finalPrompt += `Conversation:\n`;
        if (history) {
            let conversation = '';
            let conversationLength = model.tokenize(finalPrompt).length + model.tokenize(`Human: ${prompt}\nAI:`).length;
            for (let i = history.length - 1; i >= 0; i--) {
                const { source, message } = history[i];
                if (conversationLength + model.tokenize(message).length > model.trainContextSize * 0.75) {
                    break;
                }
                conversation = `${source === 'ai' ? 'AI' : 'Human'}: ${message}\n${conversation}`;
                conversationLength += model.tokenize(`${source === 'ai' ? 'AI' : 'Human'}: ${message}\n`).length;
            }
            finalPrompt += conversation;
        }
        finalPrompt += `Human: ${prompt}\nAI:`;
        return finalPrompt;
    }
    /**
     * Get the vector for some content.
     * @param embeddingContext The embedding context.
     * @param content The content.
     * @returns A promise that resolves to the vector.
     * @ignore
     */
    async getVector(embeddingContext, content) {
        const vector = (await embeddingContext.getEmbeddingFor(this.cleanText(content))).vector;
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
