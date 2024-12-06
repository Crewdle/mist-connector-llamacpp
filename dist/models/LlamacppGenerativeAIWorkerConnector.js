import { rmSync } from 'fs';
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
     * The workflow ID.
     * @ignore
     */
    workflowId;
    /**
     * The Llama engine.
     * @ignore
     */
    static engine;
    /**
     * The models.
     * @ignore
     */
    static models = new Map();
    /**
     * The embedding context.
     * @ignore
     */
    static embeddingContext;
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
    /**
     * Get the VRAM state.
     * @returns The total and available VRAM.
     */
    static async getVramState() {
        if (!LlamacppGenerativeAIWorkerConnector.engine) {
            return {
                total: 0,
                available: 0,
            };
        }
        const vramState = await LlamacppGenerativeAIWorkerConnector.engine.getVramState();
        return {
            total: vramState.total,
            available: vramState.free,
        };
    }
    /**
     * Get the Llama engine.
     * @returns A promise that resolves with the Llama engine.
     * @ignore
     */
    static async getEngine() {
        if (LlamacppGenerativeAIWorkerConnector.engine) {
            console.log('Using existing Llama engine');
            return LlamacppGenerativeAIWorkerConnector.engine;
        }
        console.log('Loading Llama engine');
        const { getLlama } = await import('node-llama-cpp');
        LlamacppGenerativeAIWorkerConnector.engine = await getLlama();
        console.log('Llama engine loaded');
        return LlamacppGenerativeAIWorkerConnector.engine;
    }
    /**
     * Get a model.
     * @param id The model ID.
     * @returns The model.
     * @ignore
     */
    static getModel(id) {
        return LlamacppGenerativeAIWorkerConnector.models.get(id);
    }
    /**
     * Set a model.
     * @param id The model ID.
     * @param model The model.
     * @ignore
     */
    static setModel(id, model) {
        LlamacppGenerativeAIWorkerConnector.models.set(id, model);
    }
    /**
     * Delete a model.
     * @param id The model ID.
     * @ignore
     */
    static deleteModel(id) {
        LlamacppGenerativeAIWorkerConnector.models.delete(id);
    }
    /**
     * Initialize the machine learning model.
     * @param workflowId The workflow ID.
     * @param models The models to initialize.
     */
    async initialize(workflowId, models) {
        const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
        for (const [modelName, modelObj] of models) {
            if (modelObj.engineType !== 'llamacpp' || !modelObj.pathName) {
                continue;
            }
            let model = LlamacppGenerativeAIWorkerConnector.getModel(modelName);
            if (!model) {
                try {
                    if (modelObj.outputType === 'vector') {
                        const modelInstance = await engine.loadModel({
                            modelPath: modelObj.pathName,
                            useMlock: false,
                        });
                        model = {
                            model: modelInstance,
                            workflows: new Set(),
                        };
                        if (!LlamacppGenerativeAIWorkerConnector.embeddingContext) {
                            LlamacppGenerativeAIWorkerConnector.embeddingContext = await modelInstance.createEmbeddingContext();
                        }
                    }
                    else {
                        const modelInstance = await engine.loadModel({
                            modelPath: modelObj.pathName,
                            useMlock: false,
                            defaultContextFlashAttention: true,
                        });
                        model = {
                            model: modelInstance,
                            workflows: new Set(),
                        };
                    }
                    model.workflows.add(workflowId);
                    LlamacppGenerativeAIWorkerConnector.setModel(modelName, model);
                    this.workflowId = workflowId;
                }
                catch (e) {
                    console.error(e);
                    rmSync(modelObj.pathName);
                    throw e;
                }
            }
            else {
                model.workflows.add(workflowId);
                LlamacppGenerativeAIWorkerConnector.setModel(modelName, model);
                this.workflowId = workflowId;
            }
        }
    }
    /**
     * Close the machine learning model.
     * @returns A promise that resolves when the model has been closed.
     */
    async close() {
        if (!this.workflowId) {
            return;
        }
        if (LlamacppGenerativeAIWorkerConnector.embeddingContext) {
            await LlamacppGenerativeAIWorkerConnector.embeddingContext.dispose();
        }
        for (const [id, model] of LlamacppGenerativeAIWorkerConnector.models) {
            model.workflows.delete(this.workflowId);
            if (model.workflows.size === 0) {
                await model.model.dispose();
                LlamacppGenerativeAIWorkerConnector.deleteModel(id);
            }
            else {
                LlamacppGenerativeAIWorkerConnector.setModel(id, model);
            }
        }
    }
    getEngineType() {
        return 'llamacpp';
    }
    /**
     * Process a job.
     * @param parameters The job parameters.
     * @returns A promise that resolves with the job result.
     */
    async processJob(parameters, options) {
        const model = LlamacppGenerativeAIWorkerConnector.getModel(options.model.id)?.model;
        if (!model) {
            throw new Error('Model not initialized');
        }
        if (options.model.outputType === 'vector') {
            if (!LlamacppGenerativeAIWorkerConnector.embeddingContext) {
                throw new Error('Embedding context not initialized');
            }
            const vector = await this.getVector(LlamacppGenerativeAIWorkerConnector.embeddingContext, parameters.prompt);
            return {
                type: 'prompt',
                output: vector,
            };
        }
        const context = await model.createContext();
        console.log('Context size', context.contextSize);
        const { prompt, functions, grammar, maxTokens, temperature, instructions } = parameters;
        const sequence = context.getSequence();
        const { LlamaChatSession } = await import('node-llama-cpp');
        const session = new LlamaChatSession({
            contextSequence: sequence,
            systemPrompt: instructions ?? this.instructions,
        });
        const startingInputTokens = sequence.tokenMeter.usedInputTokens;
        const startingOutputTokens = sequence.tokenMeter.usedOutputTokens;
        this.setupSession(session, parameters);
        let promptOptions = {
            functions: functions ? await this.getFunctions(functions) : undefined,
            grammar: undefined,
        };
        if (grammar) {
            if (grammar === 'json' || grammar === 'json_arr') {
                promptOptions.functions = undefined;
                promptOptions.grammar = await (await LlamacppGenerativeAIWorkerConnector.getEngine()).getGrammarFor(grammar);
            }
            else if (grammar !== 'default') {
                promptOptions.grammar = await (await LlamacppGenerativeAIWorkerConnector.getEngine()).createGrammar({
                    grammar,
                });
            }
        }
        const output = await session.prompt(prompt, {
            maxTokens: maxTokens ?? this.maxTokens,
            temperature: temperature ?? this.temperature,
            ...promptOptions,
        });
        const inputTokens = sequence.tokenMeter.usedInputTokens - startingInputTokens;
        const outputTokens = sequence.tokenMeter.usedOutputTokens - startingOutputTokens;
        session.dispose();
        sequence.dispose();
        await context.dispose();
        return {
            type: 'prompt',
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
        const model = LlamacppGenerativeAIWorkerConnector.getModel(options.model.id)?.model;
        if (!model) {
            throw new Error('Model not initialized');
        }
        if (options.model.outputType === 'vector') {
            throw new Error('Vector output type not supported for streaming');
        }
        const context = await model.createContext();
        console.log('Context size', context.contextSize);
        const { prompt, functions, grammar, maxTokens, temperature, instructions } = parameters;
        const sequence = context.getSequence();
        const { LlamaChatSession } = await import('node-llama-cpp');
        const session = new LlamaChatSession({
            contextSequence: sequence,
            systemPrompt: instructions ?? this.instructions,
        });
        const startingInputTokens = sequence.tokenMeter.usedInputTokens;
        const startingOutputTokens = sequence.tokenMeter.usedOutputTokens;
        this.setupSession(session, parameters);
        let promptOptions = {
            functions: functions ? await this.getFunctions(functions) : undefined,
            grammar: undefined,
        };
        if (grammar) {
            if (grammar === 'json' || grammar === 'json_arr') {
                promptOptions.functions = undefined;
                promptOptions.grammar = await (await LlamacppGenerativeAIWorkerConnector.getEngine()).getGrammarFor(grammar);
            }
            else if (grammar !== 'default') {
                promptOptions.grammar = await (await LlamacppGenerativeAIWorkerConnector.getEngine()).createGrammar({
                    grammar,
                });
            }
        }
        const textEmitter = new EventEmitter();
        session.prompt(prompt, {
            maxTokens: maxTokens ?? this.maxTokens,
            temperature: temperature ?? this.temperature,
            ...promptOptions,
            onTextChunk: (text) => {
                textEmitter.emit('text', text);
            },
        }).then(() => {
            textEmitter.emit('text', undefined);
        }).catch(e => {
            console.error(e);
            textEmitter.emit('text', undefined);
        });
        while (true) {
            const text = await new Promise((resolve) => textEmitter.once('text', resolve));
            if (text === undefined) {
                break;
            }
            yield {
                type: 'prompt',
                output: text,
                inputTokens: sequence.tokenMeter.usedInputTokens - startingInputTokens,
                outputTokens: sequence.tokenMeter.usedOutputTokens - startingOutputTokens,
            };
        }
        session.dispose();
        sequence.dispose();
        await context.dispose();
    }
    setupSession(session, parameters) {
        const { instructions, history } = parameters;
        const chatHistory = [{
                type: 'system',
                text: instructions ?? this.instructions,
            }];
        if (history) {
            for (const item of history) {
                if (item.source === 'ai') {
                    chatHistory.push({
                        type: 'model',
                        response: [item.message],
                    });
                }
                if (item.source === 'human') {
                    chatHistory.push({
                        type: 'user',
                        text: item.message,
                    });
                }
            }
        }
        session.setChatHistory(chatHistory);
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
    /**
     * Get the functions object.
     * @param functions The functions to process
     * @returns A promise that resolves to the functions object
     */
    async getFunctions(functions) {
        const { defineChatSessionFunction } = await import('node-llama-cpp');
        let functionsObj = {};
        Array.from(functions?.entries()).map(([name, func]) => {
            if (!functionsObj) {
                return;
            }
            if (func.params) {
                functionsObj[name] = defineChatSessionFunction({
                    description: func.description,
                    params: {
                        type: 'object',
                        properties: {
                            ...func.params,
                        }
                    },
                    handler(params) {
                        return func.callback(params);
                    },
                });
            }
            else {
                functionsObj[name] = defineChatSessionFunction({
                    description: func.description,
                    handler() {
                        return func.callback();
                    }
                });
            }
        });
        return functionsObj;
    }
}
