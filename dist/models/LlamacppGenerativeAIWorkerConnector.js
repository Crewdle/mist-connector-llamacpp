"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __await = (this && this.__await) || function (v) { return this instanceof __await ? (this.v = v, this) : new __await(v); }
var __asyncGenerator = (this && this.__asyncGenerator) || function (thisArg, _arguments, generator) {
    if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
    var g = generator.apply(thisArg, _arguments || []), i, q = [];
    return i = {}, verb("next"), verb("throw"), verb("return", awaitReturn), i[Symbol.asyncIterator] = function () { return this; }, i;
    function awaitReturn(f) { return function (v) { return Promise.resolve(v).then(f, reject); }; }
    function verb(n, f) { if (g[n]) { i[n] = function (v) { return new Promise(function (a, b) { q.push([n, v, a, b]) > 1 || resume(n, v); }); }; if (f) i[n] = f(i[n]); } }
    function resume(n, v) { try { step(g[n](v)); } catch (e) { settle(q[0][3], e); } }
    function step(r) { r.value instanceof __await ? Promise.resolve(r.value.v).then(fulfill, reject) : settle(q[0][2], r); }
    function fulfill(value) { resume("next", value); }
    function reject(value) { resume("throw", value); }
    function settle(f, v) { if (f(v), q.shift(), q.length) resume(q[0][0], q[0][1]); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.LlamacppGenerativeAIWorkerConnector = void 0;
const events_1 = require("events");
/**
 * The Llamacpp machine learning connector.
 */
class LlamacppGenerativeAIWorkerConnector {
    /**
     * The constructor.
     * @param vectorDatabaseConnector The vector database connector constructor.
     */
    constructor(vectorDatabaseConnector, options) {
        this.options = options;
        /**
         * The instructions.
         * @ignore
         */
        this.instructions = 'Keep the answer short and concise.';
        /**
         * The sentences.
         * @ignore
         */
        this.sentences = [];
        this.vectorDatabase = new vectorDatabaseConnector();
    }
    /**
     * Initialize the machine learning model.
     * @param llmModel The path to the LLM model.
     * @param similarityModel The path to the similarity model.
     */
    initialize(llmModel, similarityModel) {
        return __awaiter(this, void 0, void 0, function* () {
            var _a, _b, _c, _d, _e, _f;
            if (!llmModel && !((_a = this.options) === null || _a === void 0 ? void 0 : _a.llmPath)) {
                throw new Error('LLM model path not provided');
            }
            if (!similarityModel && !((_b = this.options) === null || _b === void 0 ? void 0 : _b.similarityPath)) {
                throw new Error('Similarity model path not provided');
            }
            const { getLlama } = yield Promise.resolve().then(() => __importStar(require('node-llama-cpp')));
            this.engine = yield getLlama();
            this.llmModel = yield this.engine.loadModel({
                modelPath: (_d = llmModel !== null && llmModel !== void 0 ? llmModel : (_c = this.options) === null || _c === void 0 ? void 0 : _c.llmPath) !== null && _d !== void 0 ? _d : 'llm.gguf',
            });
            this.similarityModel = yield this.engine.loadModel({
                modelPath: (_f = similarityModel !== null && similarityModel !== void 0 ? similarityModel : (_e = this.options) === null || _e === void 0 ? void 0 : _e.similarityPath) !== null && _f !== void 0 ? _f : 'similarity.gguf',
            });
        });
    }
    /**
     * Add content to the machine learning model.
     * @param content The content to add.
     * @returns A promise that resolves when the content has been added.
     */
    addContent(content) {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.similarityModel) {
                throw new Error('Model not initialized');
            }
            this.embeddingContext = yield this.similarityModel.createEmbeddingContext();
            const sentences = this.splitIntoSentences(content);
            this.sentences.push(...sentences);
            const embeddings = yield Promise.all(sentences.map(sentence => this.getVector(sentence)));
            this.vectorDatabase.insert(embeddings);
            yield this.embeddingContext.dispose();
            this.embeddingContext = undefined;
        });
    }
    /**
     * Prompt the machine learning model.
     * @param prompt The prompt to use.
     * @returns An async generator that yields the responses.
     */
    processJob(job) {
        return __asyncGenerator(this, arguments, function* processJob_1() {
            if (!this.llmModel) {
                throw new Error('Model not initialized');
            }
            if (job.parameters.jobType !== 0) {
                return yield __await(void 0);
            }
            const { LlamaChatSession } = yield __await(Promise.resolve().then(() => __importStar(require('node-llama-cpp'))));
            const prompt = job.parameters.prompt;
            const context = yield __await(this.getContext(prompt));
            if (!this.chatContext) {
                this.chatContext = yield __await(this.llmModel.createContext());
            }
            if (!this.chatSession) {
                this.chatSession = new LlamaChatSession({
                    contextSequence: this.chatContext.getSequence(),
                });
            }
            const tokenEmitter = new events_1.EventEmitter();
            this.chatSession.prompt(`Instructions: ${this.instructions}, Context: ${context}, Prompt: ${prompt}`, {
                maxTokens: 1024,
                temperature: 0.5,
                onToken: (token) => __awaiter(this, void 0, void 0, function* () {
                    tokenEmitter.emit('token', token);
                })
            });
            while (true) {
                const token = yield __await(new Promise((resolve) => tokenEmitter.once('token', resolve)));
                if (!token)
                    break;
                yield yield __await({
                    id: job.id,
                    status: 'Completed',
                    result: {
                        jobType: 0,
                        output: this.llmModel.detokenize(token),
                    },
                });
            }
        });
    }
    /**
     * Get the context for a prompt.
     * @param prompt The prompt.
     * @returns A promise that resolves to the context.
     * @ignore
     */
    getContext(prompt) {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.similarityModel) {
                throw new Error('Model not initialized');
            }
            this.embeddingContext = yield this.similarityModel.createEmbeddingContext();
            const embedding = yield this.getVector(prompt);
            const context = this.vectorDatabase.search(embedding, 5).map(index => this.sentences.slice(index - 2, index + 3).join(' '));
            yield this.embeddingContext.dispose();
            return context.join(' ');
        });
    }
    /**
     * Get the vector for some content.
     * @param content The content.
     * @returns A promise that resolves to the embedding.
     * @ignore
     */
    getVector(content) {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.llmModel) {
                throw new Error('Model not initialized');
            }
            if (!this.embeddingContext) {
                throw new Error('Embedding context not initialized');
            }
            return (yield this.embeddingContext.getEmbeddingFor(content)).vector;
        });
    }
    /**
     * Split text into sentences.
     * @param text The text to split.
     * @returns The sentences.
     * @ignore
     */
    splitIntoSentences(text) {
        var _a;
        return ((_a = text.match(/[^\.!\?]+[\.!\?]+/g)) === null || _a === void 0 ? void 0 : _a.filter(sentence => sentence.trim().length > 0)) || [];
    }
}
exports.LlamacppGenerativeAIWorkerConnector = LlamacppGenerativeAIWorkerConnector;
