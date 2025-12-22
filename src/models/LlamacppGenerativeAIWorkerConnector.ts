import { existsSync, unlinkSync, readdirSync, readFileSync, statSync } from 'fs';
import { join, resolve } from 'path';
import { homedir, platform, arch, release, totalmem, freemem, cpus } from 'os';
import { execSync } from 'child_process';

import { EventEmitter } from 'events';

import { Llama, LlamaEmbeddingContext, LlamaContext, ChatHistoryItem, LlamaChatSession, LlamaContextSequence, resolveModelFile } from 'node-llama-cpp';

import type { GenerativeAIEngineType, GenerativeAIModelOutputType, IGenerativeAIModel, IGenerativeAIWorkerConnector, IGenerativeAIWorkerOptions, GenerativeAIWorkerConnectorParameters, GenerativeAIWorkerConnectorResult, IGenerativeAIPromptWorkerConnectorParameters, IGenerativeAIWorkerConnectorPromptResult, GenerativeAIWorkerConnectorTypes, PromptFunction } from '@crewdle/web-sdk-types';

import { ILlamacppGenerativeAIWorkerOptions } from './LlamacppGenerativeAIWorkerOptions';
import { ILlamacppGenerativeAIWorkerModel } from './LlamacppGenerativeAIWorkerModel';

import officeParser from 'officeparser';
import robot from '@jitsi/robotjs';
import screenshotDesktop from 'screenshot-desktop';

/**
 * The Llamacpp machine learning connector.
 */
export class LlamacppGenerativeAIWorkerConnector implements IGenerativeAIWorkerConnector {
  /**
   * The instructions.
   * @ignore
   */
  private instructions = `The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it will use concepts stored in memory that have very similar weights.`;

  /**
   * The maximum number of tokens to generate.
   * @ignore
   */
  private maxTokens = 4096;

  /**
   * The temperature.
   * @ignore
   */
  private temperature = 1;

  /**
   * The workflow ID.
   * @ignore
   */
  private workflowId?: string;

  /**
   * The Llama engine.
   * @ignore
   */
  private static engine?: Llama;

  /**
   * The models.
   * @ignore
   */
  private static models: Map<string, {
    model: ILlamacppGenerativeAIWorkerModel,
    context?: LlamaContext
  }> = new Map();

  /**
   * The embedding context.
   * @ignore
   */
  private static embeddingContext?: LlamaEmbeddingContext;

  /**
   * The base folder.
   * @ignore
   */
  private baseFolder?: string;

  /**
   * The constructor.
   * @param options The options.
   */
  constructor(
    private apiKey?: string,
    private options?: ILlamacppGenerativeAIWorkerOptions,
  ) {
    if (this.options?.instructions) {
      this.instructions = this.options.instructions;
    }
    if (this.options?.maxTokens) {
      this.maxTokens = this.options.maxTokens;
    }
    if (this.options?.temperature) {
      this.temperature = this.options.temperature;
    }

    this.baseFolder = this.options?.baseFolder;
  }

  /**
   * Get the VRAM state.
   * @returns The total and available VRAM.
   */
  static async getVramState(): Promise<{ total: number, available: number }> {
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
  private static async getEngine(): Promise<Llama> {
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
  private static getModel(id: string): ILlamacppGenerativeAIWorkerModel | undefined {
    return LlamacppGenerativeAIWorkerConnector.models.get(id)?.model;
  }

  /**
   * Get a model context.
   * @param id The model ID.
   * @returns The model context.
   * @ignore
   */
  private static getContext(id: string): LlamaContext | undefined {
    return LlamacppGenerativeAIWorkerConnector.models.get(id)?.context;
  }

  /**
   * Set a model.
   * @param id The model ID.
   * @param model The model.
   * @ignore
   */
  private static setModel(id: string, model: ILlamacppGenerativeAIWorkerModel): void {
    let existingModel = LlamacppGenerativeAIWorkerConnector.models.get(id);
    if (existingModel) {
      existingModel.model = model;
    } else {
      existingModel = {
        model,
      };
    }

    LlamacppGenerativeAIWorkerConnector.models.set(id, existingModel);
  }

  /**
   * Set a model context.
   * @param id The model ID.
   * @param context The model context.
   * @ignore
   */
  private static setContext(id: string, context: LlamaContext | undefined): void {
    const existingModel = LlamacppGenerativeAIWorkerConnector.models.get(id);
    if (existingModel) {
      existingModel.context = context;
      LlamacppGenerativeAIWorkerConnector.models.set(id, existingModel);
    }
  }

  /**
   * Delete a model.
   * @param id The model ID.
   * @ignore
   */
  private static deleteModel(id: string): void {
    LlamacppGenerativeAIWorkerConnector.models.delete(id);
  }

  /**
   * Initialize the machine learning model.
   * @param workflowId The workflow ID.
   * @param models The models to initialize.
   */
  async initialize(workflowId: string, models: Map<string, IGenerativeAIModel>): Promise<void> {
    if (!this.baseFolder) {
      throw new Error('Base folder not set');
    }

    const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
    console.log('Engine', engine.gpu, engine.supportsGpuOffloading);
    const vramState = await engine.getVramState();
    console.log('VRAM state', vramState.total, vramState.free, vramState.used, vramState.unifiedSize);
    for (const [modelName, modelObj] of models) {
      if (modelObj.engineType !== 'llamacpp') {
        continue;
      }
      let model = LlamacppGenerativeAIWorkerConnector.getModel(modelName);
      if (!model) {
        try {
          modelObj.pathName = await resolveModelFile(
            modelObj.sourceUrl,
            this.baseFolder
          );

          if (modelObj.outputType === 'vector' as GenerativeAIModelOutputType.Vector) {
            const modelInstance = await engine.loadModel({
              modelPath: modelObj.pathName,
              useMlock: false,
            });
            model = {
              pathName: modelObj.pathName,
              outputType: modelObj.outputType,
              model: modelInstance,
              workflows: new Set(),
            }
            if (!LlamacppGenerativeAIWorkerConnector.embeddingContext) {
              LlamacppGenerativeAIWorkerConnector.embeddingContext = await modelInstance.createEmbeddingContext();
            }
          } else {
            model = {
              pathName: modelObj.pathName,
              outputType: modelObj.outputType,
              workflows: new Set(),
            }
          }

          model.workflows.add(workflowId);
          LlamacppGenerativeAIWorkerConnector.setModel(modelName, model);
          this.workflowId = workflowId;
        } catch (e) {
          console.error(e);
          const modelPath = `${this.baseFolder}/${modelObj.id}.gguf`;
          if (existsSync(modelPath)) {
            unlinkSync(modelPath);
          }
          throw e;
        }
      } else {
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
  async close(): Promise<void> {
    if (!this.workflowId) {
      return;
    }
    if (LlamacppGenerativeAIWorkerConnector.embeddingContext) {
      await LlamacppGenerativeAIWorkerConnector.embeddingContext.dispose();
      LlamacppGenerativeAIWorkerConnector.embeddingContext = undefined;
    }
    for (const [id, model] of LlamacppGenerativeAIWorkerConnector.models) {
      model.model.workflows.delete(this.workflowId);
      if (model.model.workflows.size === 0) {
        if (model.model.model) {
          await model.model.model.dispose();
        }
        LlamacppGenerativeAIWorkerConnector.deleteModel(id);
      } else {
        LlamacppGenerativeAIWorkerConnector.setModel(id, model.model);
      }
    }
  }

  getEngineType(): GenerativeAIEngineType {
    return 'llamacpp' as GenerativeAIEngineType;
  }

  /**
   * Process a job.
   * @param parameters The job parameters.
   * @returns A promise that resolves with the job result.
   */
  async processJob(parameters: GenerativeAIWorkerConnectorParameters, options: IGenerativeAIWorkerOptions): Promise<GenerativeAIWorkerConnectorResult> {
    const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
    const modelObj = LlamacppGenerativeAIWorkerConnector.getModel(options.model.id);

    if (!modelObj) {
      throw new Error('Model not initialized');
    }

    let model = modelObj.model;
    if (!model) {
      for (const [id, m] of LlamacppGenerativeAIWorkerConnector.models.entries()) {
        if (m.model.outputType === 'text' as GenerativeAIModelOutputType.Text && m.model.model) {
          console.log('Disposing model', id);
          await m.model.model.dispose();
          m.model.model = undefined;
          LlamacppGenerativeAIWorkerConnector.setModel(id, m.model);
        }
      }
      console.log('Loading model', options.model.id);
      model = await engine.loadModel({
        modelPath: modelObj.pathName,
        useMlock: false,
        defaultContextFlashAttention: true,
      });
      modelObj.model = model;
      LlamacppGenerativeAIWorkerConnector.setModel(options.model.id, modelObj);
    }

    if (options.model.outputType === 'vector' as GenerativeAIModelOutputType.Vector) {
      if (!LlamacppGenerativeAIWorkerConnector.embeddingContext) {
        throw new Error('Embedding context not initialized');
      }
      const vector = await this.getVector(LlamacppGenerativeAIWorkerConnector.embeddingContext, parameters.prompt);
      return {
        type: 'prompt' as GenerativeAIWorkerConnectorTypes,
        output: vector,
      };
    }

    let context: LlamaContext | undefined;
    let sequence: LlamaContextSequence | undefined;
    let session: LlamaChatSession | undefined;

    try {
      context = LlamacppGenerativeAIWorkerConnector.getContext(options.model.id);
      if (!context) {
        context = await model.createContext({
          sequences: options.sequences,
        });
        LlamacppGenerativeAIWorkerConnector.setContext(options.model.id, context);
      }
      console.log('Context size', context.contextSize);
      const { prompt, functions, reasoning, grammar, maxTokens, temperature, instructions } = parameters;
      sequence = context.getSequence();
      const { LlamaChatSession } = await import('node-llama-cpp');
      session = new LlamaChatSession({
        contextSequence: sequence,
        systemPrompt: instructions ?? this.instructions,
      });

      const startingInputTokens = sequence.tokenMeter.usedInputTokens;
      const startingOutputTokens = sequence.tokenMeter.usedOutputTokens;

      this.setupSession(session, parameters);

      let promptOptions: {
        functions?: {[key: string]: any},
        grammar?: any,
      } = {
        functions: await this.getFunctions(functions ?? new Map()),
        grammar: undefined,
      }
      if (grammar) {
        if (grammar === 'json' || grammar === 'json_arr') {
          promptOptions.functions = undefined;
          const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
          promptOptions.grammar = await engine.getGrammarFor(grammar);
        } else if (grammar === 'if_else') {
          promptOptions.functions = undefined;
          const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
          promptOptions.grammar = await engine.createGrammarForJsonSchema({
            type: 'object',
            properties: {
              'condition_result': {
                type: 'boolean'
              }
            },
            required: ['result'],
            additionalProperties: false,
          });
        } else if (grammar !== 'default') {
          promptOptions.functions = undefined;
          const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
          promptOptions.grammar = await engine.createGrammarForJsonSchema(grammar);
        }
      }

      let output = '';
      let finalPrompt = '';
      if (reasoning) {
        let thinking = '';
        thinking += await session.prompt(`Analyze the following question and explain the steps needed to answer it without actually providing the final answer. Explain the reasoning process, key considerations, and possible strategies to find the answer.\nQuestion: ${prompt}`, {
          maxTokens: maxTokens ?? this.maxTokens,
          temperature: temperature ?? this.temperature,
          ...promptOptions,
        });

        if (thinking.length < 500) {
          thinking += await session.prompt(`Analyze the following question and explain the steps needed to answer it without actually providing the final answer. Explain the reasoning process, key considerations, and possible strategies to find the answer.\nInitial thoughts:${thinking}\nQuestion: ${prompt}`, {
            maxTokens: maxTokens ?? this.maxTokens,
            temperature: temperature ?? this.temperature,
            ...promptOptions,
          });
        }

        output += `<think>${thinking}</think>\n\n`;
        finalPrompt += `<think>${thinking}</think>\n\n`;
      }
      finalPrompt += prompt;
      output += await session.prompt(finalPrompt, {
        maxTokens: maxTokens ?? this.maxTokens,
        temperature: temperature ?? this.temperature,
        ...promptOptions,
      });

      const inputTokens = sequence.tokenMeter.usedInputTokens - startingInputTokens;
      const outputTokens = sequence.tokenMeter.usedOutputTokens - startingOutputTokens

      session.dispose();
      sequence.dispose();
      if (context.sequencesLeft === (options.sequences ?? 1)) {
        await context.dispose();
        LlamacppGenerativeAIWorkerConnector.setContext(options.model.id, undefined);
      }

      return {
        type: 'prompt' as GenerativeAIWorkerConnectorTypes,
        output,
        inputTokens,
        outputTokens,
      };
    } catch (e) {
      if (session && !session.disposed) {
        session.dispose();
      }
      if (sequence && !sequence.disposed) {
        sequence.dispose();
      }
      if (context && !context.disposed && context.sequencesLeft === (options.sequences ?? 1)) {
        await context.dispose();
        LlamacppGenerativeAIWorkerConnector.setContext(options.model.id, undefined);
      }
      console.error(e);
      throw e;
    }
  }

  /**
   * Stream a job.
   * @param parameters The job parameters.
   * @returns An async generator that yields the responses.
   */
  async *processJobStream(parameters: IGenerativeAIPromptWorkerConnectorParameters, options: IGenerativeAIWorkerOptions): AsyncGenerator<IGenerativeAIWorkerConnectorPromptResult> {
    const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
    const modelObj = LlamacppGenerativeAIWorkerConnector.getModel(options.model.id);

    if (!modelObj) {
      throw new Error('Model not initialized');
    }

    let model = modelObj.model;
    if (!model) {
      for (const [id, m] of LlamacppGenerativeAIWorkerConnector.models.entries()) {
        if (m.model.outputType === 'text' as GenerativeAIModelOutputType.Text && m.model.model) {
          console.log('Disposing model', id);
          await m.model.model.dispose();
          m.model.model = undefined;
          LlamacppGenerativeAIWorkerConnector.setModel(id, m.model);
        }
      }
      console.log('Loading model', options.model.id);
      model = await engine.loadModel({
        modelPath: modelObj.pathName,
        useMlock: false,
        defaultContextFlashAttention: true,
      });
      modelObj.model = model;
      LlamacppGenerativeAIWorkerConnector.setModel(options.model.id, modelObj);
    }

    if (options.model.outputType === 'vector' as GenerativeAIModelOutputType.Vector) {
      throw new Error('Vector output type not supported for streaming');
    }

    let context: LlamaContext | undefined;
    let sequence: LlamaContextSequence | undefined;
    let session: LlamaChatSession | undefined;

    try {
      context = LlamacppGenerativeAIWorkerConnector.getContext(options.model.id);
      if (!context) {
        context = await model.createContext({
          sequences: options.sequences,
        });
        LlamacppGenerativeAIWorkerConnector.setContext(options.model.id, context);
      }
      console.log('Context size', context.contextSize);
      const { prompt, functions, reasoning, grammar, maxTokens, temperature, instructions } = parameters;
      sequence = context.getSequence();
      const { LlamaChatSession } = await import('node-llama-cpp');
      session = new LlamaChatSession({
        contextSequence: sequence,
        systemPrompt: instructions ?? this.instructions,
      });

      const startingInputTokens = sequence.tokenMeter.usedInputTokens;
      const startingOutputTokens = sequence.tokenMeter.usedOutputTokens;

      this.setupSession(session, parameters);

      let promptOptions: {
        functions?: {[key: string]: any},
        grammar?: any,
      } = {
        functions: await this.getFunctions(functions ?? new Map()),
        grammar: undefined,
      }
      if (grammar) {
        if (grammar === 'json' || grammar === 'json_arr') {
          promptOptions.functions = undefined;
          const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
          promptOptions.grammar = await engine.getGrammarFor(grammar);
        } else if (grammar === 'if_else') {
          promptOptions.functions = undefined;
          const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
          promptOptions.grammar = await engine.createGrammarForJsonSchema({
            type: 'object',
            properties: {
              'condition_result': {
                type: 'boolean'
              }
            },
            required: ['result'],
            additionalProperties: false,
          });
        } else if (grammar !== 'default') {
          promptOptions.functions = undefined;
          const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
          promptOptions.grammar = await engine.createGrammarForJsonSchema(grammar);
        }
      }

      const textEmitter = new EventEmitter();

      const runPrompt = (finalPrompt: string) => {
        session!.prompt(finalPrompt, {
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
      };

      if (reasoning) {
        let thinking = '';
        session.prompt(`Analyze the following question and explain the steps needed to answer it without actually providing the final answer. Explain the reasoning process, key considerations, and possible strategies to find the answer.\nQuestion: ${prompt}`, {
          maxTokens: maxTokens ?? this.maxTokens,
          temperature: temperature ?? this.temperature,
          onTextChunk: (text) => {
            if (thinking === '') {
              textEmitter.emit('text', `<think>${text}`);
            } else {
              textEmitter.emit('text', text);
            }
            thinking += text;
          },
        }).then(() => {
          if (thinking.length < 500) {
            session!.prompt(`Analyze the following question and explain the steps needed to answer it without actually providing the final answer. Explain the reasoning process, key considerations, and possible strategies to find the answer.\nInitial thoughts:${thinking}\nQuestion: ${prompt}`, {
              maxTokens: maxTokens ?? this.maxTokens,
              temperature: temperature ?? this.temperature,
              onTextChunk: (text) => {
                textEmitter.emit('text', text);
                thinking += text;
              },
            }).then(() => {
              textEmitter.emit('text', '</think>\n\n');
              runPrompt(`<think>${thinking}</think>\n\n${prompt}`);
            }).catch(e => {
              console.error(e);
              textEmitter.emit('text', undefined);
            });
          } else {
            textEmitter.emit('text', '</think>\n\n');
            runPrompt(`<think>${thinking}</think>\n\n${prompt}`);
          }
        }).catch(e => {
          console.error(e);
          textEmitter.emit('text', undefined);
        });
      } else {
        runPrompt(prompt);
      }

      while (true) {
        const text = await new Promise<string>((resolve) => textEmitter.once('text', resolve));

        if (text === undefined) {
          break;
        }

        yield {
          type: 'prompt' as GenerativeAIWorkerConnectorTypes,
          output: text,
          inputTokens: sequence.tokenMeter.usedInputTokens - startingInputTokens,
          outputTokens: sequence.tokenMeter.usedOutputTokens - startingOutputTokens,
        };
      }

      session.dispose();
      sequence.dispose();
      if (context.sequencesLeft === (options.sequences ?? 1)) {
        await context.dispose();
        LlamacppGenerativeAIWorkerConnector.setContext(options.model.id, undefined);
      }
    } catch (e) {
      if (session && !session.disposed) {
        session.dispose();
      }
      if (sequence && !sequence.disposed) {
        sequence.dispose();
      }
      if (context && !context.disposed && context.sequencesLeft === (options.sequences ?? 1)) {
        await context.dispose();
        LlamacppGenerativeAIWorkerConnector.setContext(options.model.id, undefined);
      }
      console.error(e);
      throw e;
    }
  }

  private setupSession(session: LlamaChatSession, parameters: GenerativeAIWorkerConnectorParameters): void {
    const { instructions, history } = parameters;

    const chatHistory: ChatHistoryItem[] = [{
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
  private async getVector(embeddingContext: LlamaEmbeddingContext, content: string): Promise<number[]> {
    const vector = (await embeddingContext.getEmbeddingFor(this.cleanText(content))).vector;
    return this.normalizeVector(vector);
  }

  /**
   * Clean the text.
   * @param text The text to clean.
   * @returns The cleaned text.
   * @ignore
   */
  private cleanText(text: string): string {
    return text.trim().toLowerCase();
  }

  /**
   * Normalize a vector.
   * @param vector The vector to normalize.
   * @returns The normalized vector.
   * @ignore
   */
  private normalizeVector(vector: readonly number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    if (Math.abs(norm - 1) < 0.01) {
      return vector.map((value) => value);
    }
    return vector.map((value) => value / norm);
  }

  /**
   * Get the functions object.
   * @param functions The functions to process
   * @returns A promise that resolves to the functions object
   */
  private async getFunctions(functions: Map<string, PromptFunction>): Promise<{[key: string]: any}> {
    const { defineChatSessionFunction } = await import('node-llama-cpp');
    let functionsObj: {[key: string]: any} = {}

    // Add built-in system functions
    functionsObj['list_directory'] = defineChatSessionFunction({
      description: 'List files and directories in a specified path',
      params: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'The directory path to list. Defaults to current working directory if not provided.',
          }
        }
      },
      handler(params: { path?: string }) {
        console.log('list_directory', params);
        try {
          const dirPath = params.path ? resolve(params.path) : process.cwd();
          const items = readdirSync(dirPath);
          const details = items.map(item => {
            const fullPath = join(dirPath, item);
            const stats = statSync(fullPath);
            return {
              name: item,
              type: stats.isDirectory() ? 'directory' : 'file',
              size: stats.size,
              modified: stats.mtime.toISOString(),
            };
          });
          return JSON.stringify({ path: dirPath, items: details });
        } catch (error: any) {
          return JSON.stringify({ error: error.message });
        }
      },
    });

    functionsObj['read_file'] = defineChatSessionFunction({
      description: 'Read the contents of a file',
      params: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'The file path to read',
          },
          encoding: {
            type: 'string',
            description: 'The file encoding. Defaults to utf8.',
          }
        }
      },
      handler(params: { path: string; encoding?: string }) {
        console.log('read_file', params);
        try {
          const filePath = resolve(params.path);
          const content = readFileSync(filePath, { encoding: (params.encoding || 'utf8') as BufferEncoding });
          return JSON.stringify({ path: filePath, content, size: content.length });
        } catch (error: any) {
          return JSON.stringify({ error: error.message });
        }
      },
    });

    functionsObj['read_document'] = defineChatSessionFunction({
      description: 'Read and parse document files (PDF, DOCX, PPTX, ODT, etc.) and extract text content',
      params: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'The document file path to read',
          }
        }
      },
      handler: async (params: { path: string }) => {
        console.log('read_document', params);
        try {
          const filePath = resolve(params.path);
          const text = await officeParser.parseOfficeAsync(filePath);
          const stats = statSync(filePath);

          return JSON.stringify({
            path: filePath,
            content: text,
            size: stats.size,
            wordCount: text.split(/\s+/).filter(word => word.length > 0).length,
            characterCount: text.length,
          });
        } catch (error: any) {
          return JSON.stringify({ error: error.message });
        }
      },
    });

    functionsObj['search_files'] = defineChatSessionFunction({
      description: 'Search for files using glob patterns (e.g., **/*.ts, src/**/*.json)',
      params: {
        type: 'object',
        properties: {
          pattern: {
            type: 'string',
            description: 'The glob pattern to match files (e.g., **/*.ts, src/**/*.json)',
          },
          basePath: {
            type: 'string',
            description: 'The base directory to search from. Defaults to current working directory.',
          }
        }
      },
      handler(params: { pattern: string; basePath?: string }) {
        console.log('search_files', params);
        try {
          const searchPath = params.basePath ? resolve(params.basePath) : process.cwd();
          const results: string[] = [];

          const searchDir = (dir: string, pattern: string) => {
            const items = readdirSync(dir);
            for (const item of items) {
              const fullPath = join(dir, item);
              try {
                const stats = statSync(fullPath);
                if (stats.isDirectory()) {
                  if (!item.startsWith('.') && item !== 'node_modules') {
                    searchDir(fullPath, pattern);
                  }
                } else {
                  const regex = new RegExp(pattern.replace(/\*/g, '.*').replace(/\?/g, '.'));
                  if (regex.test(fullPath)) {
                    results.push(fullPath);
                  }
                }
              } catch (e) {
                // Skip inaccessible files
              }
            }
          };

          searchDir(searchPath, params.pattern);
          return JSON.stringify({ pattern: params.pattern, basePath: searchPath, results, count: results.length });
        } catch (error: any) {
          return JSON.stringify({ error: error.message });
        }
      },
    });

    functionsObj['grep'] = defineChatSessionFunction({
      description: 'Search for text patterns in files using regex',
      params: {
        type: 'object',
        properties: {
          pattern: {
            type: 'string',
            description: 'The regex pattern to search for',
          },
          path: {
            type: 'string',
            description: 'The file or directory path to search in',
          },
          filePattern: {
            type: 'string',
            description: 'Optional glob pattern to filter files (e.g., *.ts)',
          }
        }
      },
      handler(params: { pattern: string; path: string; filePattern?: string }) {
        console.log('grep', params);
        try {
          const searchPath = resolve(params.path);
          const regex = new RegExp(params.pattern, 'gi');
          const matches: Array<{ file: string; line: number; content: string }> = [];

          const searchFile = (filePath: string) => {
            try {
              const content = readFileSync(filePath, 'utf8');
              const lines = content.split('\n');
              lines.forEach((line, index) => {
                if (regex.test(line)) {
                  matches.push({
                    file: filePath,
                    line: index + 1,
                    content: line.trim(),
                  });
                }
              });
            } catch (e) {
              // Skip unreadable files
            }
          };

          const searchDir = (dir: string) => {
            const items = readdirSync(dir);
            for (const item of items) {
              const fullPath = join(dir, item);
              try {
                const stats = statSync(fullPath);
                if (stats.isDirectory()) {
                  if (!item.startsWith('.') && item !== 'node_modules') {
                    searchDir(fullPath);
                  }
                } else {
                  if (!params.filePattern || new RegExp(params.filePattern.replace(/\*/g, '.*')).test(item)) {
                    searchFile(fullPath);
                  }
                }
              } catch (e) {
                // Skip inaccessible files
              }
            }
          };

          const stats = statSync(searchPath);
          if (stats.isDirectory()) {
            searchDir(searchPath);
          } else {
            searchFile(searchPath);
          }

          return JSON.stringify({ pattern: params.pattern, path: searchPath, matches, count: matches.length });
        } catch (error: any) {
          return JSON.stringify({ error: error.message });
        }
      },
    });

    functionsObj['get_working_directory'] = defineChatSessionFunction({
      description: 'Get the current working directory',
      handler() {
        console.log('get_working_directory');
        try {
          const cwd = process.cwd();
          return JSON.stringify({ cwd, home: homedir() });
        } catch (error: any) {
          return JSON.stringify({ error: error.message });
        }
      },
    });

    functionsObj['get_system_info'] = defineChatSessionFunction({
      description: 'Get system information including OS, platform, memory, and CPU details',
      handler() {
        console.log('get_system_info');
        try {
          const info = {
            platform: platform(),
            arch: arch(),
            release: release(),
            hostname: execSync('hostname', { encoding: 'utf8' }).trim(),
            cpus: cpus().map(cpu => ({
              model: cpu.model,
              speed: cpu.speed,
            })),
            cpuCount: cpus().length,
            totalMemory: totalmem(),
            freeMemory: freemem(),
            memoryUsage: process.memoryUsage(),
            uptime: process.uptime(),
          };
          return JSON.stringify(info);
        } catch (error: any) {
          return JSON.stringify({ error: error.message });
        }
      },
    });

    functionsObj['execute_command'] = defineChatSessionFunction({
      description: 'Execute shell commands or AppleScript (on macOS). Use shell commands for general operations or AppleScript for macOS-specific automation.',
      params: {
        type: 'object',
        properties: {
          command: {
            type: 'string',
            description: 'The command to execute',
          },
          type: {
            type: 'string',
            description: 'Command type: "shell" (default) or "applescript" (macOS only)',
          }
        }
      },
      handler(params: { command: string; type?: string }) {
        console.log('execute_command', params);
        try {
          const cmdType = params.type || 'shell';
          let result: string;
          let statusCode: number = 0;

          if (cmdType === 'applescript' && platform() === 'darwin') {
            try {
              result = execSync(`osascript -e '${params.command.replace(/'/g, "'\\''")}'`, { encoding: 'utf8' });
            } catch (e: any) {
              result = e.message;
              statusCode = e.status || 1;
            }
          } else {
            try {
              result = execSync(params.command, { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });
            } catch (e: any) {
              result = e.message;
              statusCode = e.status || 1;
            }
          }

          return JSON.stringify({ output: result, statusCode, command: params.command });
        } catch (error: any) {
          return JSON.stringify({ error: error.message });
        }
      },
    });

    functionsObj['click_at_position'] = defineChatSessionFunction({
      description: 'Click the mouse at specific screen coordinates',
      params: {
        type: 'object',
        properties: {
          x: {
            type: 'number',
            description: 'X coordinate',
          },
          y: {
            type: 'number',
            description: 'Y coordinate',
          },
          button: {
            type: 'string',
            description: 'Mouse button: "left" (default), "right", or "middle"',
          },
          double: {
            type: 'boolean',
            description: 'Whether to double click',
          }
        }
      },
      handler: async (params: { x: number; y: number; button?: string; double?: boolean }) => {
        console.log('click_at_position', params);
        try {
          robot.moveMouse(params.x, params.y);

          const button = params.button || 'left';
          const double = params.double || false;

          robot.mouseClick(button, double);

          return JSON.stringify({
            action: 'click',
            x: params.x,
            y: params.y,
            button,
            double
          });
        } catch (error: any) {
          return JSON.stringify({ error: error.message });
        }
      },
    });

    functionsObj['type_text'] = defineChatSessionFunction({
      description: 'Type text at the current cursor position',
      params: {
        type: 'object',
        properties: {
          text: {
            type: 'string',
            description: 'The text to type',
          }
        }
      },
      handler: async (params: { text: string }) => {
        console.log('type_text', params);
        try {
          robot.typeString(params.text);

          return JSON.stringify({
            action: 'type',
            text: params.text
          });
        } catch (error: any) {
          return JSON.stringify({ error: error.message });
        }
      },
    });

    functionsObj['take_screenshot'] = defineChatSessionFunction({
      description: 'Take a screenshot of the entire screen or active display. Returns base64 encoded image data.',
      handler: async () => {
        console.log('take_screenshot');
        try {
          const imgBuffer = await screenshotDesktop();
          const base64 = imgBuffer.toString('base64');

          return JSON.stringify({
            action: 'screenshot',
            format: 'png',
            encoding: 'base64',
            data: base64,
            size: imgBuffer.length,
          });
        } catch (error: any) {
          return JSON.stringify({ error: error.message });
        }
      },
    });

    functionsObj['press_keys'] = defineChatSessionFunction({
      description: 'Press keyboard shortcuts/key combinations. Use + to combine keys (e.g., "command+c", "control+shift+t")',
      params: {
        type: 'object',
        properties: {
          shortcut: {
            type: 'string',
            description: 'Key combination separated by + (e.g., "command+c", "control+alt+delete")',
          }
        }
      },
      handler: async (params: { shortcut: string }) => {
        console.log('press_keys', params);
        try {
          const keys = params.shortcut.toLowerCase().split('+').map(k => k.trim());

          // Handle modifiers and regular keys
          const modifiers: string[] = [];
          let regularKey: string | undefined;

          for (const key of keys) {
            if (['command', 'cmd', 'control', 'ctrl', 'shift', 'alt', 'option'].includes(key)) {
              modifiers.push(key === 'cmd' ? 'command' : key === 'ctrl' ? 'control' : key === 'option' ? 'alt' : key);
            } else {
              regularKey = key;
            }
          }

          if (regularKey && modifiers.length > 0) {
            robot.keyTap(regularKey, modifiers);
          } else if (regularKey) {
            robot.keyTap(regularKey);
          } else if (modifiers.length > 0) {
            // Just press the modifier
            robot.keyTap(modifiers[0]);
          }

          return JSON.stringify({
            action: 'press_keys',
            shortcut: params.shortcut
          });
        } catch (error: any) {
          return JSON.stringify({ error: error.message });
        }
      },
    });

    // Add user-provided functions
    Array.from(functions?.entries()).map(([name, func]) => {
      if (func.type !== 'function') {
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
      } else {
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
