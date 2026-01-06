import { readdirSync, readFileSync, statSync } from 'fs';
import { join, resolve } from 'path';
import { homedir, platform, arch, release, totalmem, freemem, cpus } from 'os';
import { execSync } from 'child_process';

import { EventEmitter } from 'events';

import { Llama, LlamaEmbeddingContext, LlamaContext, ChatHistoryItem, LlamaChatSession, LlamaContextSequence, resolveModelFile, readGgufFileInfo, GgufInsights } from 'node-llama-cpp';

import type { GenerativeAIEngineType, GenerativeAIModelOutputType, IGenerativeAIModel, IGenerativeAIWorkerConnector, IGenerativeAIWorkerOptions, GenerativeAIWorkerConnectorParameters, GenerativeAIWorkerConnectorResult, IGenerativeAIPromptWorkerConnectorParameters, IGenerativeAIWorkerConnectorPromptResult, GenerativeAIWorkerConnectorTypes, PromptFunction } from '@crewdle/web-sdk-types';

import { ILlamacppGenerativeAIWorkerOptions } from './LlamacppGenerativeAIWorkerOptions';
import { ILlamacppGenerativeAIWorkerModel } from './LlamacppGenerativeAIWorkerModel';

import officeParser from 'officeparser';

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
   * Microsoft Graph OAuth tokens.
   * @ignore
   */
  private static microsoftGraphTokens?: {
    accessToken: string;
    refreshToken?: string;
    tokenType?: string;
    expiresIn?: number;
    expiresAt?: number; // Timestamp when token expires
    scope?: string;
  };

  /**
   * Callback to refresh Microsoft Graph access tokens.
   * @ignore
   */
  private static refreshTokensCallback?: (refreshToken: string) => Promise<{
    accessToken: string;
    refreshToken?: string;
    tokenType?: string;
    expiresIn?: number;
    scope?: string;
  }>;

  /**
   * Callback to search the web.
   * @ignore
   */
  private static browserCallback?: (url: string) => Promise<string>;

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

  static getModels(): Map<string, ILlamacppGenerativeAIWorkerModel> {
    const models = new Map<string, ILlamacppGenerativeAIWorkerModel>();
    for (const [id, model] of LlamacppGenerativeAIWorkerConnector.models) {
      models.set(id, {
        pathName: model.model.pathName ?? '',
        outputType: model.model.outputType ?? 'text' as GenerativeAIModelOutputType,
        downloaded: model.model.downloaded ?? false,
        totalSize: model.model.totalSize ?? undefined,
        downloadedSize: model.model.downloadedSize ?? undefined,
        workflows: new Set(),
      });
    }
    return models;
  }

  static async downloadModel(id: string, baseFolder: string): Promise<void> {
    const model = LlamacppGenerativeAIWorkerConnector.getModel(id);
    if (!model) {
      return;
    }
    const pathName = await resolveModelFile(
      model.pathName,
      {
        directory: baseFolder,
        onProgress: (status: { totalSize: number, downloadedSize: number }) => {
          model.totalSize = status.totalSize;
          model.downloadedSize = status.downloadedSize;
          LlamacppGenerativeAIWorkerConnector.setModel(id, model);
        },
      }
    );
    model.pathName = pathName;
    model.downloaded = true;
    LlamacppGenerativeAIWorkerConnector.setModel(id, model);
  }

  static async loadModel(id: string): Promise<void> {
    const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
    const modelObj = LlamacppGenerativeAIWorkerConnector.getModel(id);

    if (!modelObj) {
      throw new Error('Model not initialized');
    }

    if (!modelObj.downloaded) {
      throw new Error('Model not downloaded');
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
      console.log('Loading model', id);
      model = await engine.loadModel({
        gpuLayers: {
          fitContext: {
            contextSize: 8192,
          },
        },
        modelPath: modelObj.pathName,
        defaultContextFlashAttention: true,
      });
      modelObj.model = model;
      console.log('Model loaded', model.gpuLayers);
      LlamacppGenerativeAIWorkerConnector.setModel(id, modelObj);
    }
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
   * Set the Microsoft Graph OAuth tokens.
   * @param tokens The OAuth tokens for Microsoft Graph.
   */
  static setMicrosoftGraphTokens(tokens: {
    accessToken: string;
    refreshToken?: string;
    tokenType?: string;
    expiresAt?: number;
    scope?: string;
  } | undefined): void {
    LlamacppGenerativeAIWorkerConnector.microsoftGraphTokens = tokens;
  }

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
  }>): void {
    LlamacppGenerativeAIWorkerConnector.refreshTokensCallback = callback;
  }

  /**
   * Set the callback to browse the web.
   * @param callback The callback function to browse the web.
   */
  static setBrowserCallback(callback: (query: string) => Promise<string>): void {
    LlamacppGenerativeAIWorkerConnector.browserCallback = callback;
  }

  /**
   * Get the Microsoft Graph OAuth tokens.
   * Automatically refreshes tokens if they are expired or about to expire (within 5 minutes).
   * @returns A promise that resolves with the OAuth tokens if set, undefined otherwise.
   */
  private static async getMicrosoftGraphTokens(): Promise<{
    accessToken: string;
    refreshToken?: string;
    tokenType?: string;
    expiresAt?: number;
    scope?: string;
  } | undefined> {
    const tokens = LlamacppGenerativeAIWorkerConnector.microsoftGraphTokens;

    if (!tokens) {
      throw new Error('Connect your Microsoft account first');
    }

    // Check if token is expired or about to expire (within 5 minutes)
    const REFRESH_THRESHOLD = 5 * 60 * 1000; // 5 minutes in milliseconds
    if (tokens.expiresAt && tokens.refreshToken) {
      const timeUntilExpiry = tokens.expiresAt - Date.now();
      if (timeUntilExpiry < REFRESH_THRESHOLD) {
        try {
          return await LlamacppGenerativeAIWorkerConnector.refreshAccessTokens();
        } catch (error) {
          console.error('Failed to refresh expired token:', error);
          // Return existing token anyway, let the API call fail if truly expired
          return tokens;
        }
      }
    }

    return tokens;
  }

  /**
   * Refresh the Microsoft Graph access tokens.
   * @returns A promise that resolves with the refreshed access tokens.
   * @throws Error if no refresh callback is set or no refresh token is available.
   */
  private static async refreshAccessTokens(): Promise<{
    accessToken: string;
    refreshToken?: string;
    tokenType?: string;
    expiresAt?: number;
    scope?: string;
  }> {
    if (!LlamacppGenerativeAIWorkerConnector.refreshTokensCallback) {
      throw new Error('No refresh tokens callback has been set. Call setRefreshTokensCallback() first.');
    }
    if (!LlamacppGenerativeAIWorkerConnector.microsoftGraphTokens?.refreshToken) {
      throw new Error('No refresh token available. Get tokens first using getAccessTokens().');
    }
    const tokens = await LlamacppGenerativeAIWorkerConnector.refreshTokensCallback(
      LlamacppGenerativeAIWorkerConnector.microsoftGraphTokens.refreshToken
    );
    LlamacppGenerativeAIWorkerConnector.microsoftGraphTokens = tokens;
    return tokens;
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
    for (const [modelName, modelObj] of models) {
      if (modelObj.engineType !== 'llamacpp') {
        continue;
      }
      let model = LlamacppGenerativeAIWorkerConnector.getModel(modelName);
      if (!model) {
        try {
          const modelMetadata = await readGgufFileInfo(modelObj.sourceUrl);

          const insights = await GgufInsights.from(modelMetadata, engine);
          const resolvedConfig = await insights.configurationResolver.resolveAndScoreConfig({
            targetGpuLayers: 'max'
          });

          if (resolvedConfig.compatibilityScore < 0.95) {
            console.log(`Incompatible: ${modelName} ${resolvedConfig.compatibilityScore * 100}%`);
            continue;
          }

          console.log(`Compatibility: ${modelName} ${resolvedConfig.compatibilityScore * 100}%`, resolvedConfig);

          modelObj.pathName = await resolveModelFile(
            modelObj.sourceUrl,
            {
              directory: this.baseFolder,
              download: false,
            }
          );
          console.log('Model path', modelObj.pathName);

          if (modelObj.outputType === 'vector' as GenerativeAIModelOutputType.Vector) {
            const modelInstance = await engine.loadModel({
              modelPath: modelObj.pathName,
              useMlock: false,
            });
            model = {
              pathName: modelObj.pathName,
              outputType: modelObj.outputType,
              model: modelInstance,
              downloaded: true,
              workflows: new Set(),
            }
            if (!LlamacppGenerativeAIWorkerConnector.embeddingContext) {
              LlamacppGenerativeAIWorkerConnector.embeddingContext = await modelInstance.createEmbeddingContext();
            }
          } else {
            model = {
              pathName: modelObj.pathName,
              outputType: modelObj.outputType,
              downloaded: true,
              workflows: new Set(),
            }
          }

          model.workflows.add(workflowId);
          LlamacppGenerativeAIWorkerConnector.setModel(modelName, model);
          this.workflowId = workflowId;
        } catch (e) {
          model = {
            pathName: modelObj.sourceUrl,
            outputType: modelObj.outputType,
            downloaded: false,
            workflows: new Set(),
          }

          model.workflows.add(workflowId);
          LlamacppGenerativeAIWorkerConnector.setModel(modelName, model);
          this.workflowId = workflowId;
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
        gpuLayers: {
          fitContext: {
            contextSize: 8192,
          },
        },
        modelPath: modelObj.pathName,
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
      console.log('Context size', context.contextSize, options.sequences);
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
        functions: await this.getFunctions(functions ?? new Map(), options.model.id),
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

      if (parameters.instructions) {
        console.log('Removing functions because of instructions', parameters.instructions);
        promptOptions.functions = undefined;
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
        gpuLayers: {
          fitContext: {
            contextSize: 8192,
          },
        },
        modelPath: modelObj.pathName,
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
      console.log('Context size', context.contextSize, options.sequences);
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
        functions: await this.getFunctions(functions ?? new Map(), options.model.id),
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

      if (parameters.instructions) {
        console.log('Removing functions because of instructions', parameters.instructions);
        promptOptions.functions = undefined;
      }
      
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
   * @param modelId The model ID to get context size from
   * @returns A promise that resolves to the functions object
   */
  private async getFunctions(functions: Map<string, PromptFunction>, modelId?: string): Promise<{[key: string]: any}> {
    const { defineChatSessionFunction } = await import('node-llama-cpp');
    let functionsObj: {[key: string]: any} = {}

    // Get context size for this model, or use conservative default
    const context = modelId ? LlamacppGenerativeAIWorkerConnector.getContext(modelId) : undefined;
    const contextSizeTokens = context?.contextSize || 8192; // Default if context not available

    // Convert tokens to approximate characters (1 token ≈ 4 characters)
    // Reserve 20% of context for function responses to leave room for conversation
    const CHARS_PER_TOKEN = 4;
    const FUNCTION_RESPONSE_PERCENTAGE = 0.20;
    const MAX_RESPONSE_SIZE = Math.floor(contextSizeTokens * CHARS_PER_TOKEN * FUNCTION_RESPONSE_PERCENTAGE);
    console.log(`Function response limit: ${MAX_RESPONSE_SIZE} chars (${contextSizeTokens} tokens context, ~${Math.floor(MAX_RESPONSE_SIZE / CHARS_PER_TOKEN)} tokens)`);

    const enforceResponseLimit = (result: string): string => {
      if (result.length > MAX_RESPONSE_SIZE) {
        try {
          const obj = JSON.parse(result);
          obj.responseTruncated = true;
          obj.originalSize = result.length;
          obj.maxAllowedSize = MAX_RESPONSE_SIZE;
          obj.warning = `Response exceeded maximum size (${MAX_RESPONSE_SIZE} chars / ~${Math.floor(MAX_RESPONSE_SIZE / CHARS_PER_TOKEN)} tokens) and was truncated`;
          return JSON.stringify(obj).substring(0, MAX_RESPONSE_SIZE);
        } catch {
          return result.substring(0, MAX_RESPONSE_SIZE) + `\n[TRUNCATED: Response too large. Max: ${MAX_RESPONSE_SIZE} chars / ~${Math.floor(MAX_RESPONSE_SIZE / CHARS_PER_TOKEN)} tokens]`;
        }
      }
      return result;
    };

    // Add built-in system functions
    functionsObj['list_directory'] = defineChatSessionFunction({
      description: 'List files and directories in a specified path. Results are paginated.',
      params: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'The directory path to list. Defaults to current working directory if not provided.',
          },
          offset: {
            type: 'number',
            description: 'Number of items to skip. Default: 0',
          },
          limit: {
            type: 'number',
            description: 'Maximum number of items to return. Default: 50',
          }
        }
      },
      handler(params: { path?: string; offset?: number; limit?: number }) {
        console.log('list_directory', params);
        try {
          const dirPath = params.path ? resolve(params.path) : process.cwd();
          const offset = params.offset || 0;
          const limit = params.limit || 50;

          const items = readdirSync(dirPath);
          const totalCount = items.length;
          const paginatedItems = items.slice(offset, offset + limit);

          const details = paginatedItems.map(item => {
            const fullPath = join(dirPath, item);
            const stats = statSync(fullPath);
            return {
              name: item,
              type: stats.isDirectory() ? 'directory' : 'file',
              size: stats.size,
              modified: stats.mtime.toISOString(),
            };
          });

          return enforceResponseLimit(JSON.stringify({
            path: dirPath,
            items: details,
            count: details.length,
            totalCount: totalCount,
            offset: offset,
            hasMore: offset + limit < totalCount
          }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['read_file'] = defineChatSessionFunction({
      description: 'Read the contents of a file. Results are paginated by lines for large files.',
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
          },
          startLine: {
            type: 'number',
            description: 'Line number to start reading from (1-based). Default: 1',
          },
          lineCount: {
            type: 'number',
            description: 'Number of lines to read. Default: 500. Set to 0 to read entire file (not recommended).',
          }
        }
      },
      handler(params: { path: string; encoding?: string; startLine?: number; lineCount?: number }) {
        console.log('read_file', params);
        try {
          const filePath = resolve(params.path);
          const content = readFileSync(filePath, { encoding: (params.encoding || 'utf8') as BufferEncoding });
          const lines = content.split('\n');
          const totalLines = lines.length;

          const startLine = Math.max(1, params.startLine || 1);
          const lineCount = params.lineCount === 0 ? totalLines : (params.lineCount || 1000);

          const startIndex = startLine - 1;
          const endIndex = Math.min(startIndex + lineCount, totalLines);
          const selectedLines = lines.slice(startIndex, endIndex);
          const paginatedContent = selectedLines.join('\n');

          return enforceResponseLimit(JSON.stringify({
            path: filePath,
            content: paginatedContent,
            totalLines: totalLines,
            startLine: startLine,
            endLine: startIndex + selectedLines.length,
            linesReturned: selectedLines.length,
            hasMore: endIndex < totalLines,
            truncated: lineCount > 0 && totalLines > lineCount
          }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['read_document'] = defineChatSessionFunction({
      description: 'Read and parse document files (PDF, DOCX, PPTX, ODT, etc.) and extract text content. Results are paginated for large documents.',
      params: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'The document file path to read',
          },
          charOffset: {
            type: 'number',
            description: 'Character offset to start reading from. Default: 0',
          },
          charLimit: {
            type: 'number',
            description: 'Maximum number of characters to return. Default: 10000. Set to 0 for entire document.',
          }
        }
      },
      handler: async (params: { path: string; charOffset?: number; charLimit?: number }) => {
        console.log('read_document', params);
        try {
          const filePath = resolve(params.path);
          const text = await officeParser.parseOfficeAsync(filePath);
          const stats = statSync(filePath);

          const charOffset = params.charOffset || 0;
          const charLimit = params.charLimit === 0 ? text.length : (params.charLimit || 10000);

          const totalChars = text.length;
          const endOffset = Math.min(charOffset + charLimit, totalChars);
          const content = text.substring(charOffset, endOffset);

          return enforceResponseLimit(JSON.stringify({
            path: filePath,
            content: content,
            size: stats.size,
            totalCharacters: totalChars,
            charOffset: charOffset,
            charEnd: endOffset,
            charactersReturned: content.length,
            wordCount: content.split(/\s+/).filter(word => word.length > 0).length,
            hasMore: endOffset < totalChars,
            truncated: charLimit > 0 && totalChars > charLimit
          }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['search_files'] = defineChatSessionFunction({
      description: 'Search for files using glob patterns (e.g., **/*.ts, src/**/*.json). Results are limited to avoid context overflow.',
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
          },
          limit: {
            type: 'number',
            description: 'Maximum number of results to return. Default: 100',
          }
        }
      },
      handler(params: { pattern: string; basePath?: string; limit?: number }) {
        console.log('search_files', params);
        try {
          const searchPath = params.basePath ? resolve(params.basePath) : process.cwd();
          const maxResults = params.limit || 100;
          const results: string[] = [];
          let totalCount = 0;

          const searchDir = (dir: string, pattern: string) => {
            if (results.length >= maxResults) return;

            const items = readdirSync(dir);
            for (const item of items) {
              if (results.length >= maxResults) break;

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
                    totalCount++;
                    if (results.length < maxResults) {
                      results.push(fullPath);
                    }
                  }
                }
              } catch (e) {
                // Skip inaccessible files
              }
            }
          };

          searchDir(searchPath, params.pattern);

          return enforceResponseLimit(JSON.stringify({
            pattern: params.pattern,
            basePath: searchPath,
            results,
            count: results.length,
            totalMatches: totalCount,
            truncated: totalCount > maxResults
          }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['grep'] = defineChatSessionFunction({
      description: 'Search for text patterns in files using regex. Results are limited to avoid context overflow.',
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
          },
          limit: {
            type: 'number',
            description: 'Maximum number of matches to return. Default: 100',
          }
        }
      },
      handler(params: { pattern: string; path: string; filePattern?: string; limit?: number }) {
        console.log('grep', params);
        try {
          const searchPath = resolve(params.path);
          const regex = new RegExp(params.pattern, 'gi');
          const maxResults = params.limit || 100;
          const matches: Array<{ file: string; line: number; content: string }> = [];
          let totalCount = 0;

          const searchFile = (filePath: string) => {
            if (matches.length >= maxResults) return;

            try {
              const content = readFileSync(filePath, 'utf8');
              const lines = content.split('\n');
              for (let index = 0; index < lines.length; index++) {
                if (matches.length >= maxResults) break;

                const line = lines[index];
                if (regex.test(line)) {
                  totalCount++;
                  if (matches.length < maxResults) {
                    matches.push({
                      file: filePath,
                      line: index + 1,
                      content: line.trim().substring(0, 200), // Truncate long lines
                    });
                  }
                }
              }
            } catch (e) {
              // Skip unreadable files
            }
          };

          const searchDir = (dir: string) => {
            if (matches.length >= maxResults) return;

            const items = readdirSync(dir);
            for (const item of items) {
              if (matches.length >= maxResults) break;

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

          return enforceResponseLimit(JSON.stringify({
            pattern: params.pattern,
            path: searchPath,
            matches,
            count: matches.length,
            totalMatches: totalCount,
            truncated: totalCount > maxResults
          }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['get_working_directory'] = defineChatSessionFunction({
      description: 'Get the current working directory',
      handler() {
        console.log('get_working_directory');
        try {
          const cwd = process.cwd();
          return enforceResponseLimit(JSON.stringify({ cwd, home: homedir() }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
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
          return enforceResponseLimit(JSON.stringify(info));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['search_web'] = defineChatSessionFunction({
      description: 'Search the web for information. Results are truncated for large result sets.',
      params: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The query to search for',
          },
          charLimit: {
            type: 'number',
            description: 'Maximum number of characters to return. Default: 15000. Set to 0 for all results.',
          }
        }
      },
      handler: async (params: { query: string; charLimit?: number }) => {
        console.log('search_web', params);
        try {
          const rawHtml = await LlamacppGenerativeAIWorkerConnector.browserCallback?.(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(params.query)}`);
          if (!rawHtml) {
            return enforceResponseLimit(JSON.stringify({ error: 'Failed to search the web' }));
          }

          const content = this.markdownSearch(rawHtml);
          const charLimit = params.charLimit === 0 ? content.length : (params.charLimit || 15000);
          const totalChars = content.length;
          const truncatedContent = content.substring(0, charLimit);

          console.log('search_web content length', content.length);

          return enforceResponseLimit(JSON.stringify({
            query: params.query,
            results: truncatedContent,
            totalCharacters: totalChars,
            charactersReturned: truncatedContent.length,
            truncated: totalChars > charLimit
          }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['fetch_url'] = defineChatSessionFunction({
      description: 'Fetch the content of a specific URL. Content is truncated for large pages.',
      params: {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'The URL to fetch',
          },
          charLimit: {
            type: 'number',
            description: 'Maximum number of characters to return. Default: 20000. Set to 0 for entire content.',
          }
        }
      },
      handler: async (params: { url: string; charLimit?: number }) => {
        console.log('fetch_url', params);
        try {
          const rawHtml = await LlamacppGenerativeAIWorkerConnector.browserCallback?.(params.url);
          if (!rawHtml) {
            return enforceResponseLimit(JSON.stringify({ error: 'Failed to fetch the URL' }));
          }

          const content = this.markdownContent(rawHtml);
          const charLimit = params.charLimit === 0 ? content.length : (params.charLimit || 20000);
          const totalChars = content.length;
          const truncatedContent = content.substring(0, charLimit);

          console.log('fetch_url content length', content.length);

          return enforceResponseLimit(JSON.stringify({
            url: params.url,
            content: truncatedContent,
            totalCharacters: totalChars,
            charactersReturned: truncatedContent.length,
            truncated: totalChars > charLimit
          }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['draft_email'] = defineChatSessionFunction({
      description: 'Draft an email',
      params: {
        type: 'object',
        properties: {
          subject: {
            type: 'string',
            description: 'The subject of the email',
          },
          body: {
            type: 'string',
            description: 'The body of the email',
          },
          toRecipients: {
            type: 'array',
            description: 'The recipients of the email',
            items: {
              type: 'string',
              description: 'The email address of the recipient',
            }
          },
          ccRecipients: {
            type: 'array',
            description: 'The CC recipients of the email',
            items: {
              type: 'string',
              description: 'The email address of the CC recipient',
            }
          },
          bccRecipients: {
            type: 'array',
            description: 'The BCC recipients of the email',
            items: {
              type: 'string',
              description: 'The email address of the BCC recipient',
            }
          },
          contentType: {
            type: 'string',
            description: 'The content type of the email body: Text or HTML. Defaults to Text.',
          }
        }
      },
      handler: async (params: { subject: string; body: string; toRecipients: string[]; ccRecipients?: string[]; bccRecipients?: string[]; contentType?: string }) => {
        console.log('draft_email', params);
        try {
          const tokens = await LlamacppGenerativeAIWorkerConnector.getMicrosoftGraphTokens();
          if (!tokens) {
            return enforceResponseLimit(JSON.stringify({ error: 'Connect your Microsoft account first' }));
          }

          const tokenType = tokens.tokenType || 'Bearer';

          const message = {
            subject: params.subject,
            body: {
              contentType: params.contentType || 'Text',
              content: params.body,
            },
            toRecipients: params.toRecipients.map(email => ({
              emailAddress: { address: email }
            })),
            ...(params.ccRecipients && params.ccRecipients.length > 0 && {
              ccRecipients: params.ccRecipients.map(email => ({
                emailAddress: { address: email }
              }))
            }),
            ...(params.bccRecipients && params.bccRecipients.length > 0 && {
              bccRecipients: params.bccRecipients.map(email => ({
                emailAddress: { address: email }
              }))
            }),
          };

          const response = await fetch('https://graph.microsoft.com/v1.0/me/messages', {
            method: 'POST',
            body: JSON.stringify(message),
            headers: {
              'Authorization': `${tokenType} ${tokens.accessToken}`,
              'Content-Type': 'application/json',
            },
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return enforceResponseLimit(JSON.stringify({
              error: 'Failed to draft email',
              status: response.status,
              details: errorData
            }));
          }

          const data = await response.json();
          return enforceResponseLimit(JSON.stringify(data));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['search_emails'] = defineChatSessionFunction({
      description: 'Search for emails with advanced filtering, pagination, and sorting options',
      params: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The search query (e.g., "meeting notes", "from:john@example.com"). Leave empty to list all emails.',
          },
          filter: {
            type: 'string',
            description: 'OData filter expression (e.g., "from/emailAddress/address eq \'john@example.com\'", "hasAttachments eq true", "importance eq \'high\'", "receivedDateTime ge 2024-01-01T00:00:00Z")',
          },
          top: {
            type: 'number',
            description: 'Maximum number of results to return (default: 25, max: 999)',
          },
          skip: {
            type: 'number',
            description: 'Number of results to skip for pagination (default: 0)',
          },
          orderBy: {
            type: 'string',
            description: 'Field to sort by with direction (e.g., "receivedDateTime desc", "subject asc", "from/emailAddress/name asc")',
          },
          select: {
            type: 'string',
            description: 'Comma-separated list of fields to return (e.g., "subject,from,receivedDateTime,bodyPreview,hasAttachments"). Leave empty for default fields.',
          }
        }
      },
      handler: async (params: { query?: string; filter?: string; top?: number; skip?: number; orderBy?: string; select?: string }) => {
        console.log('search_emails', params);
        try {
          const tokens = await LlamacppGenerativeAIWorkerConnector.getMicrosoftGraphTokens();
          if (!tokens) {
            return enforceResponseLimit(JSON.stringify({ error: 'Connect your Microsoft account first' }));
          }

          const tokenType = tokens.tokenType || 'Bearer';

          // Build query parameters
          const queryParams: string[] = [];

          if (params.query) {
            queryParams.push(`$search="${params.query}"`);
          }

          if (params.filter) {
            queryParams.push(`$filter=${encodeURIComponent(params.filter)}`);
          }

          if (params.top) {
            queryParams.push(`$top=${Math.min(params.top, 999)}`);
          } else {
            queryParams.push('$top=25');
          }

          if (params.skip) {
            queryParams.push(`$skip=${params.skip}`);
          }

          if (params.orderBy) {
            queryParams.push(`$orderby=${encodeURIComponent(params.orderBy)}`);
          }

          if (params.select) {
            queryParams.push(`$select=${encodeURIComponent(params.select)}`);
          }

          const url = `https://graph.microsoft.com/v1.0/me/messages?${queryParams.join('&')}`;

          const headers: Record<string, string> = {
            'Authorization': `${tokenType} ${tokens.accessToken}`,
            'Content-Type': 'application/json',
          };

          // Add ConsistencyLevel header if using $search
          if (params.query) {
            headers['ConsistencyLevel'] = 'eventual';
          }

          const response = await fetch(url, { headers });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return enforceResponseLimit(JSON.stringify({
              error: 'Failed to search emails',
              status: response.status,
              details: errorData
            }));
          }

          const data = await response.json();
          return enforceResponseLimit(JSON.stringify(data));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['get_email'] = defineChatSessionFunction({
      description: 'Get the full content of a specific email by its ID. Body content is truncated for large emails.',
      params: {
        type: 'object',
        properties: {
          messageId: {
            type: 'string',
            description: 'The ID of the email message to retrieve',
          },
          includeBody: {
            type: 'boolean',
            description: 'Whether to include the full email body (default: true)',
          },
          bodyCharLimit: {
            type: 'number',
            description: 'Maximum number of characters to return from email body. Default: 10000. Set to 0 for entire body.',
          }
        }
      },
      handler: async (params: { messageId: string; includeBody?: boolean; bodyCharLimit?: number }) => {
        console.log('get_email', params);
        try {
          const tokens = await LlamacppGenerativeAIWorkerConnector.getMicrosoftGraphTokens();
          if (!tokens) {
            return enforceResponseLimit(JSON.stringify({ error: 'Connect your Microsoft account first' }));
          }

          const tokenType = tokens.tokenType || 'Bearer';
          const includeBody = params.includeBody !== false;

          const url = includeBody
            ? `https://graph.microsoft.com/v1.0/me/messages/${params.messageId}`
            : `https://graph.microsoft.com/v1.0/me/messages/${params.messageId}?$select=id,subject,from,toRecipients,ccRecipients,receivedDateTime,sentDateTime,hasAttachments,importance,isRead,bodyPreview`;

          const response = await fetch(url, {
            headers: {
              'Authorization': `${tokenType} ${tokens.accessToken}`,
              'Content-Type': 'application/json',
            },
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return enforceResponseLimit(JSON.stringify({
              error: 'Failed to get email',
              status: response.status,
              details: errorData
            }));
          }

          const data = await response.json();

          // Truncate body if needed
          if (includeBody && data.body && data.body.content) {
            const bodyCharLimit = params.bodyCharLimit === 0 ? data.body.content.length : (params.bodyCharLimit || 10000);
            const totalChars = data.body.content.length;
            const originalBody = data.body.content;

            if (totalChars > bodyCharLimit) {
              data.body.content = originalBody.substring(0, bodyCharLimit);
              data.bodyTruncated = true;
              data.totalBodyCharacters = totalChars;
              data.bodyCharactersReturned = bodyCharLimit;
            }
          }

          return enforceResponseLimit(JSON.stringify(data));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['reply_to_email'] = defineChatSessionFunction({
      description: 'Create a draft reply to an email',
      params: {
        type: 'object',
        properties: {
          messageId: {
            type: 'string',
            description: 'The ID of the email message to reply to',
          },
          body: {
            type: 'string',
            description: 'The body of the reply',
          },
          contentType: {
            type: 'string',
            description: 'The content type of the reply body: Text or HTML. Defaults to Text.',
          },
          replyAll: {
            type: 'boolean',
            description: 'Whether to reply to all recipients (default: false)',
          }
        }
      },
      handler: async (params: { messageId: string; body: string; contentType?: string; replyAll?: boolean }) => {
        console.log('reply_to_email', params);
        try {
          const tokens = await LlamacppGenerativeAIWorkerConnector.getMicrosoftGraphTokens();
          if (!tokens) {
            return enforceResponseLimit(JSON.stringify({ error: 'Connect your Microsoft account first' }));
          }

          const tokenType = tokens.tokenType || 'Bearer';

          const endpoint = params.replyAll
            ? `https://graph.microsoft.com/v1.0/me/messages/${params.messageId}/createReplyAll`
            : `https://graph.microsoft.com/v1.0/me/messages/${params.messageId}/createReply`;

          const replyData = {
            comment: params.body,
          };

          const response = await fetch(endpoint, {
            method: 'POST',
            body: JSON.stringify(replyData),
            headers: {
              'Authorization': `${tokenType} ${tokens.accessToken}`,
              'Content-Type': 'application/json',
            },
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return enforceResponseLimit(JSON.stringify({
              error: 'Failed to create reply draft',
              status: response.status,
              details: errorData
            }));
          }

          const data = await response.json();
          return enforceResponseLimit(JSON.stringify({
            success: true,
            message: 'Reply draft created successfully',
            draftId: data.id,
            draft: data
          }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['find_meeting_times'] = defineChatSessionFunction({
      description: 'Find available meeting times for attendees within a date range',
      params: {
        type: 'object',
        properties: {
          durationMinutes: {
            type: 'number',
            description: 'Duration of the meeting in minutes (e.g., 30, 60, 90)',
          },
          startDate: {
            type: 'string',
            description: 'Start date/time for the search window in ISO 8601 format (e.g., "2024-01-15T09:00:00")',
          },
          endDate: {
            type: 'string',
            description: 'End date/time for the search window in ISO 8601 format (e.g., "2024-01-20T17:00:00")',
          },
          timeZone: {
            type: 'string',
            description: 'IANA time zone identifier (e.g., "America/New_York", "Europe/London"). Defaults to UTC.',
          },
          maxCandidates: {
            type: 'number',
            description: 'Maximum number of time suggestions to return (default: 10)',
          }
        }
      },
      handler: async (params: {
        durationMinutes: number;
        startDate: string;
        endDate: string;
        timeZone?: string;
        maxCandidates?: number;
      }) => {
        console.log('find_meeting_times', params);
        try {
          const tokens = await LlamacppGenerativeAIWorkerConnector.getMicrosoftGraphTokens();
          if (!tokens) {
            return enforceResponseLimit(JSON.stringify({ error: 'Connect your Microsoft account first' }));
          }

          const tokenType = tokens.tokenType || 'Bearer';
          const timeZone = params.timeZone || 'UTC';

          const requestBody = {
            timeConstraint: {
              timeslots: [
                {
                  start: {
                    dateTime: params.startDate,
                    timeZone: timeZone
                  },
                  end: {
                    dateTime: params.endDate,
                    timeZone: timeZone
                  }
                }
              ]
            },
            meetingDuration: `PT${params.durationMinutes}M`,
            maxCandidates: params.maxCandidates || 10,
            returnSuggestionReasons: true
          };

          const response = await fetch('https://graph.microsoft.com/v1.0/me/findMeetingTimes', {
            method: 'POST',
            body: JSON.stringify(requestBody),
            headers: {
              'Authorization': `${tokenType} ${tokens.accessToken}`,
              'Content-Type': 'application/json',
              'Prefer': `outlook.timezone="${timeZone}"`
            },
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return enforceResponseLimit(JSON.stringify({
              error: 'Failed to find meeting times',
              status: response.status,
              details: errorData
            }));
          }

          const data = await response.json();
          return enforceResponseLimit(JSON.stringify(data));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['create_meeting'] = defineChatSessionFunction({
      description: 'Create/schedule a meeting or calendar event',
      params: {
        type: 'object',
        properties: {
          subject: {
            type: 'string',
            description: 'The subject/title of the meeting',
          },
          body: {
            type: 'string',
            description: 'The body/description of the meeting',
          },
          startDateTime: {
            type: 'string',
            description: 'Start date/time in ISO 8601 format (e.g., "2024-01-15T14:00:00")',
          },
          endDateTime: {
            type: 'string',
            description: 'End date/time in ISO 8601 format (e.g., "2024-01-15T15:00:00")',
          },
          timeZone: {
            type: 'string',
            description: 'IANA time zone identifier (e.g., "America/New_York", "Europe/London"). Defaults to UTC.',
          },
          attendees: {
            type: 'array',
            description: 'Email addresses of required attendees',
            items: {
              type: 'string',
              description: 'Email address of an attendee',
            }
          },
          optionalAttendees: {
            type: 'array',
            description: 'Email addresses of optional attendees',
            items: {
              type: 'string',
              description: 'Email address of an optional attendee',
            }
          },
          location: {
            type: 'string',
            description: 'The location of the meeting (physical location or online meeting URL)',
          },
          isOnlineMeeting: {
            type: 'boolean',
            description: 'Whether to create an online meeting (Teams/Skype). Default: false',
          },
          onlineMeetingProvider: {
            type: 'string',
            description: 'Online meeting provider: teamsForBusiness or skypeForBusiness. Default: teamsForBusiness',
          },
          contentType: {
            type: 'string',
            description: 'Body content type: Text or HTML. Default: Text',
          }
        }
      },
      handler: async (params: {
        subject: string;
        body?: string;
        startDateTime: string;
        endDateTime: string;
        timeZone?: string;
        attendees?: string[];
        optionalAttendees?: string[];
        location?: string;
        isOnlineMeeting?: boolean;
        onlineMeetingProvider?: string;
        contentType?: string;
      }) => {
        console.log('create_meeting', params);
        try {
          const tokens = await LlamacppGenerativeAIWorkerConnector.getMicrosoftGraphTokens();
          if (!tokens) {
            return enforceResponseLimit(JSON.stringify({ error: 'Connect your Microsoft account first' }));
          }

          const tokenType = tokens.tokenType || 'Bearer';
          const timeZone = params.timeZone || 'UTC';

          const event = {
            subject: params.subject,
            body: {
              contentType: params.contentType || 'Text',
              content: params.body || ''
            },
            start: {
              dateTime: params.startDateTime,
              timeZone: timeZone
            },
            end: {
              dateTime: params.endDateTime,
              timeZone: timeZone
            },
            ...(params.location && {
              location: {
                displayName: params.location
              }
            }),
            ...(params.attendees && params.attendees.length > 0 && {
              attendees: [
                ...params.attendees.map(email => ({
                  type: 'required',
                  emailAddress: {
                    address: email
                  }
                })),
                ...(params.optionalAttendees || []).map(email => ({
                  type: 'optional',
                  emailAddress: {
                    address: email
                  }
                }))
              ]
            }),
            ...(params.isOnlineMeeting && {
              isOnlineMeeting: true,
              onlineMeetingProvider: params.onlineMeetingProvider || 'teamsForBusiness'
            })
          };

          const response = await fetch('https://graph.microsoft.com/v1.0/me/events', {
            method: 'POST',
            body: JSON.stringify(event),
            headers: {
              'Authorization': `${tokenType} ${tokens.accessToken}`,
              'Content-Type': 'application/json',
            },
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return enforceResponseLimit(JSON.stringify({
              error: 'Failed to create meeting',
              status: response.status,
              details: errorData
            }));
          }

          const data = await response.json();
          return enforceResponseLimit(JSON.stringify(data));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['get_calendar_events'] = defineChatSessionFunction({
      description: 'Get calendar events within a date range',
      params: {
        type: 'object',
        properties: {
          startDateTime: {
            type: 'string',
            description: 'Start date/time in ISO 8601 format (e.g., "2024-01-15T00:00:00")',
          },
          endDateTime: {
            type: 'string',
            description: 'End date/time in ISO 8601 format (e.g., "2024-01-20T23:59:59")',
          },
          timeZone: {
            type: 'string',
            description: 'IANA time zone identifier (e.g., "America/New_York", "Europe/London"). Defaults to UTC.',
          },
          top: {
            type: 'number',
            description: 'Maximum number of events to return (default: 50, max: 999)',
          },
          orderBy: {
            type: 'string',
            description: 'Field to sort by with direction (e.g., "start/dateTime asc", "start/dateTime desc")',
          },
          select: {
            type: 'string',
            description: 'Comma-separated list of fields to return (e.g., "subject,start,end,attendees,location,isOnlineMeeting")',
          }
        }
      },
      handler: async (params: {
        startDateTime: string;
        endDateTime: string;
        timeZone?: string;
        top?: number;
        orderBy?: string;
        select?: string;
      }) => {
        console.log('get_calendar_events', params);
        try {
          const tokens = await LlamacppGenerativeAIWorkerConnector.getMicrosoftGraphTokens();
          if (!tokens) {
            return enforceResponseLimit(JSON.stringify({ error: 'Connect your Microsoft account first' }));
          }

          const tokenType = tokens.tokenType || 'Bearer';
          const timeZone = params.timeZone || 'UTC';

          // Build query parameters
          const queryParams: string[] = [];

          // Add date filter
          queryParams.push(`$filter=start/dateTime ge '${params.startDateTime}' and end/dateTime le '${params.endDateTime}'`);

          if (params.top) {
            queryParams.push(`$top=${Math.min(params.top, 999)}`);
          } else {
            queryParams.push('$top=50');
          }

          if (params.orderBy) {
            queryParams.push(`$orderby=${encodeURIComponent(params.orderBy)}`);
          } else {
            queryParams.push('$orderby=start/dateTime asc');
          }

          if (params.select) {
            queryParams.push(`$select=${encodeURIComponent(params.select)}`);
          }

          const url = `https://graph.microsoft.com/v1.0/me/calendarView?${queryParams.join('&')}`;

          const response = await fetch(url, {
            headers: {
              'Authorization': `${tokenType} ${tokens.accessToken}`,
              'Content-Type': 'application/json',
              'Prefer': `outlook.timezone="${timeZone}"`
            },
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return enforceResponseLimit(JSON.stringify({
              error: 'Failed to get calendar events',
              status: response.status,
              details: errorData
            }));
          }

          const data = await response.json();
          return enforceResponseLimit(JSON.stringify(data));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['update_meeting'] = defineChatSessionFunction({
      description: 'Update/change an existing meeting or calendar event',
      params: {
        type: 'object',
        properties: {
          eventId: {
            type: 'string',
            description: 'The ID of the event to update',
          },
          subject: {
            type: 'string',
            description: 'The new subject/title of the meeting',
          },
          body: {
            type: 'string',
            description: 'The new body/description of the meeting',
          },
          startDateTime: {
            type: 'string',
            description: 'New start date/time in ISO 8601 format (e.g., "2024-01-15T14:00:00")',
          },
          endDateTime: {
            type: 'string',
            description: 'New end date/time in ISO 8601 format (e.g., "2024-01-15T15:00:00")',
          },
          timeZone: {
            type: 'string',
            description: 'IANA time zone identifier (e.g., "America/New_York", "Europe/London")',
          },
          location: {
            type: 'string',
            description: 'The new location of the meeting',
          },
          attendees: {
            type: 'array',
            description: 'Updated list of required attendees (email addresses)',
            items: {
              type: 'string',
              description: 'Email address of an attendee',
            }
          },
          optionalAttendees: {
            type: 'array',
            description: 'Updated list of optional attendees (email addresses)',
            items: {
              type: 'string',
              description: 'Email address of an optional attendee',
            }
          },
          contentType: {
            type: 'string',
            description: 'Body content type: Text or HTML',
          }
        }
      },
      handler: async (params: {
        eventId: string;
        subject?: string;
        body?: string;
        startDateTime?: string;
        endDateTime?: string;
        timeZone?: string;
        location?: string;
        attendees?: string[];
        optionalAttendees?: string[];
        contentType?: string;
      }) => {
        console.log('update_meeting', params);
        try {
          const tokens = await LlamacppGenerativeAIWorkerConnector.getMicrosoftGraphTokens();
          if (!tokens) {
            return enforceResponseLimit(JSON.stringify({ error: 'Connect your Microsoft account first' }));
          }

          const tokenType = tokens.tokenType || 'Bearer';

          // Build update object with only provided fields
          const updates: any = {};

          if (params.subject !== undefined) {
            updates.subject = params.subject;
          }

          if (params.body !== undefined) {
            updates.body = {
              contentType: params.contentType || 'Text',
              content: params.body
            };
          }

          if (params.startDateTime !== undefined && params.endDateTime !== undefined) {
            const timeZone = params.timeZone || 'UTC';
            updates.start = {
              dateTime: params.startDateTime,
              timeZone: timeZone
            };
            updates.end = {
              dateTime: params.endDateTime,
              timeZone: timeZone
            };
          }

          if (params.location !== undefined) {
            updates.location = {
              displayName: params.location
            };
          }

          if (params.attendees !== undefined) {
            updates.attendees = [
              ...params.attendees.map(email => ({
                type: 'required',
                emailAddress: {
                  address: email
                }
              })),
              ...(params.optionalAttendees || []).map(email => ({
                type: 'optional',
                emailAddress: {
                  address: email
                }
              }))
            ];
          }

          const response = await fetch(`https://graph.microsoft.com/v1.0/me/events/${params.eventId}`, {
            method: 'PATCH',
            body: JSON.stringify(updates),
            headers: {
              'Authorization': `${tokenType} ${tokens.accessToken}`,
              'Content-Type': 'application/json',
            },
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return enforceResponseLimit(JSON.stringify({
              error: 'Failed to update meeting',
              status: response.status,
              details: errorData
            }));
          }

          const data = await response.json();
          return enforceResponseLimit(JSON.stringify(data));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['cancel_meeting'] = defineChatSessionFunction({
      description: 'Cancel/delete a meeting or calendar event',
      params: {
        type: 'object',
        properties: {
          eventId: {
            type: 'string',
            description: 'The ID of the event to cancel/delete',
          },
          sendCancellation: {
            type: 'boolean',
            description: 'Whether to send cancellation notifications to attendees (default: true)',
          },
          comment: {
            type: 'string',
            description: 'Optional cancellation message to include in the notification to attendees',
          }
        }
      },
      handler: async (params: {
        eventId: string;
        sendCancellation?: boolean;
        comment?: string;
      }) => {
        console.log('cancel_meeting', params);
        try {
          const tokens = await LlamacppGenerativeAIWorkerConnector.getMicrosoftGraphTokens();
          if (!tokens) {
            return enforceResponseLimit(JSON.stringify({ error: 'Connect your Microsoft account first' }));
          }

          const tokenType = tokens.tokenType || 'Bearer';
          const sendCancellation = params.sendCancellation !== false;

          let response;

          if (sendCancellation && params.comment) {
            // Use cancel endpoint with comment
            response = await fetch(`https://graph.microsoft.com/v1.0/me/events/${params.eventId}/cancel`, {
              method: 'POST',
              body: JSON.stringify({
                comment: params.comment
              }),
              headers: {
                'Authorization': `${tokenType} ${tokens.accessToken}`,
                'Content-Type': 'application/json',
              },
            });
          } else {
            // Use delete endpoint
            response = await fetch(`https://graph.microsoft.com/v1.0/me/events/${params.eventId}`, {
              method: 'DELETE',
              headers: {
                'Authorization': `${tokenType} ${tokens.accessToken}`,
              },
            });
          }

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return enforceResponseLimit(JSON.stringify({
              error: 'Failed to cancel meeting',
              status: response.status,
              details: errorData
            }));
          }

          return enforceResponseLimit(JSON.stringify({
            success: true,
            message: sendCancellation
              ? 'Meeting canceled and attendees notified'
              : 'Meeting deleted'
          }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    // Windows privacy permission helpers (only works on Windows)
    const PERMISSION_TYPES = ['microphone', 'camera', 'location', 'notifications', 'account_info', 'contacts', 'calendar', 'phone_calls', 'call_history', 'email', 'tasks', 'messaging', 'radios', 'bluetooth', 'documents', 'pictures', 'videos', 'file_system'] as const;
    const SCOPE_TYPES = ['user', 'system'] as const;

    const PERMISSION_PATHS: Record<string, Record<string, string>> = {
      'microphone': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone'
      },
      'camera': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\webcam',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\webcam'
      },
      'location': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\location',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\location'
      },
      'notifications': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\userNotificationListener',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\userNotificationListener'
      },
      'account_info': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\userAccountInformation',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\userAccountInformation'
      },
      'contacts': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\contacts',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\contacts'
      },
      'calendar': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\appointments',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\appointments'
      },
      'phone_calls': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\phoneCall',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\phoneCall'
      },
      'call_history': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\phoneCallHistory',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\phoneCallHistory'
      },
      'email': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\email',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\email'
      },
      'tasks': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\userDataTasks',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\userDataTasks'
      },
      'messaging': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\chat',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\chat'
      },
      'radios': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\radios',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\radios'
      },
      'bluetooth': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\bluetoothSync',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\bluetoothSync'
      },
      'documents': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\documentsLibrary',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\documentsLibrary'
      },
      'pictures': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\picturesLibrary',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\picturesLibrary'
      },
      'videos': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\videosLibrary',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\videosLibrary'
      },
      'file_system': {
        'user': 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\broadFileSystemAccess',
        'system': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\broadFileSystemAccess'
      },
    };

    functionsObj['list_apps_with_permission'] = defineChatSessionFunction({
      description: 'List all apps with a specific Windows privacy permission. Use scope "system" if user-level check shows nothing. Results are limited to avoid context overflow.',
      params: {
        type: 'object',
        properties: {
          permission: {
            type: 'string',
            description: 'Permission type',
            enum: PERMISSION_TYPES
          },
          scope: {
            type: 'string',
            description: 'Check user-level (current user only) or system-level (all users) permissions. Default: "user"',
            enum: SCOPE_TYPES
          },
          limit: {
            type: 'number',
            description: 'Maximum number of apps to return. Default: 50',
          }
        },
        required: ['permission']
      } as any,
      handler(params?: { permission: string; scope?: string; limit?: number }) {
        console.log('list_apps_with_permission', params);
        try {
          if (!params) {
            return enforceResponseLimit(JSON.stringify({ error: 'Invalid parameters' }));
          }

          if (platform() !== 'win32') {
            return enforceResponseLimit(JSON.stringify({ error: 'This function only works on Windows' }));
          }

          const scope = params.scope || 'user';
          const maxResults = params.limit || 50;
          const paths = PERMISSION_PATHS[params.permission];
          if (!paths) {
            return enforceResponseLimit(JSON.stringify({ error: 'Invalid permission type' }));
          }

          const path = paths[scope];
          if (!path) {
            return enforceResponseLimit(JSON.stringify({ error: 'Invalid scope. Use "user" or "system"' }));
          }

          const command = `reg query "${path}"`;
          const output = execSync(command, { encoding: 'utf8' });

          const lines = output.split('\n');
          const apps: Array<{ name: string; path: string }> = [];
          let totalCount = 0;

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('HKEY_')) {
              const parts = trimmed.split('\\');
              const appName = parts[parts.length - 1];
              if (appName && !Object.keys(PERMISSION_PATHS).some(key => appName === key || appName === 'webcam')) {
                totalCount++;
                if (apps.length < maxResults) {
                  apps.push({
                    name: appName,
                    path: trimmed
                  });
                }
              }
            }
          }

          return enforceResponseLimit(JSON.stringify({
            permission: params.permission,
            scope: scope,
            apps: apps,
            count: apps.length,
            totalApps: totalCount,
            truncated: totalCount > maxResults
          }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['check_app_permission'] = defineChatSessionFunction({
      description: 'Check if an app has a specific Windows privacy permission. Use scope "system" if user-level check shows nothing. Common app names: msedge (Edge), chrome.exe, firefox.exe',
      params: {
        type: 'object',
        properties: {
          appName: {
            type: 'string',
            description: 'App name (e.g., "msedge" for Edge)',
          },
          permission: {
            type: 'string',
            description: 'Permission type',
            enum: PERMISSION_TYPES
          },
          scope: {
            type: 'string',
            description: 'Check user-level (current user only) or system-level (all users) permissions. Default: "user"',
            enum: SCOPE_TYPES
          }
        },
        required: ['appName', 'permission']
      } as any,
      handler(params?: { appName: string; permission: string; scope?: string }) {
        console.log('check_app_permission', params);
        try {
          if (!params) {
            return enforceResponseLimit(JSON.stringify({ error: 'Invalid parameters' }));
          }

          if (platform() !== 'win32') {
            return enforceResponseLimit(JSON.stringify({ error: 'This function only works on Windows' }));
          }

          const scope = params.scope || 'user';
          const paths = PERMISSION_PATHS[params.permission];
          if (!paths) {
            return enforceResponseLimit(JSON.stringify({ error: 'Invalid permission type' }));
          }

          const path = paths[scope];
          if (!path) {
            return enforceResponseLimit(JSON.stringify({ error: 'Invalid scope. Use "user" or "system"' }));
          }

          const appPath = `${path}\\${params.appName}`;
          const command = `reg query "${appPath}" /v Value`;

          try {
            const output = execSync(command, { encoding: 'utf8' });
            const lines = output.split('\n');
            const valueLine = lines.find(line => line.trim().startsWith('Value'));

            if (valueLine) {
              const parts = valueLine.trim().split(/\s+/);
              const value = parts[parts.length - 1];

              return enforceResponseLimit(JSON.stringify({
                app: params.appName,
                permission: params.permission,
                scope: scope,
                hasAccess: value === 'Allow',
                value: value,
                message: value === 'Allow' ? `App has ${params.permission} access` : value === 'Deny' ? `App is blocked from ${params.permission}` : 'Permission not set'
              }));
            }
          } catch (error: any) {
            if (error.message.includes('unable to find')) {
              return enforceResponseLimit(JSON.stringify({
                app: params.appName,
                permission: params.permission,
                scope: scope,
                hasAccess: false,
                message: `App not found in ${params.permission} permissions`
              }));
            }
            throw error;
          }

          return enforceResponseLimit(JSON.stringify({ error: 'Could not determine permission' }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['enable_app_permission'] = defineChatSessionFunction({
      description: 'Enable a Windows privacy permission for an app. Use scope "system" to fix system-level blocks (requires admin).',
      params: {
        type: 'object',
        properties: {
          appName: {
            type: 'string',
            description: 'App name (e.g., "msedge" for Edge)',
          },
          permission: {
            type: 'string',
            description: 'Permission type',
            enum: PERMISSION_TYPES
          },
          scope: {
            type: 'string',
            description: 'Modify user-level or system-level permissions. System requires admin. Default: "user"',
            enum: SCOPE_TYPES
          }
        },
        required: ['appName', 'permission']
      } as any,
      handler(params?: { appName: string; permission: string; scope?: string }) {
        console.log('enable_app_permission', params);
        try {
          if (!params) {
            return enforceResponseLimit(JSON.stringify({ error: 'Invalid parameters' }));
          }

          if (platform() !== 'win32') {
            return enforceResponseLimit(JSON.stringify({ error: 'This function only works on Windows' }));
          }

          const scope = params.scope || 'user';
          const paths = PERMISSION_PATHS[params.permission];
          if (!paths) {
            return enforceResponseLimit(JSON.stringify({ error: 'Invalid permission type' }));
          }

          const path = paths[scope];
          if (!path) {
            return enforceResponseLimit(JSON.stringify({ error: 'Invalid scope. Use "user" or "system"' }));
          }

          const appPath = `${path}\\${params.appName}`;
          const command = `reg add "${appPath}" /v Value /t REG_SZ /d Allow /f`;
          execSync(command, { encoding: 'utf8' });

          return enforceResponseLimit(JSON.stringify({
            success: true,
            app: params.appName,
            permission: params.permission,
            scope: scope,
            message: `${params.permission} access enabled for app at ${scope} level`
          }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
        }
      },
    });

    functionsObj['disable_app_permission'] = defineChatSessionFunction({
      description: 'Disable a Windows privacy permission for an app',
      params: {
        type: 'object',
        properties: {
          appName: {
            type: 'string',
            description: 'App name (e.g., "msedge" for Edge)',
          },
          permission: {
            type: 'string',
            description: 'Permission type',
            enum: PERMISSION_TYPES
          },
          scope: {
            type: 'string',
            description: 'Modify user-level or system-level permissions. System requires admin. Default: "user"',
            enum: SCOPE_TYPES
          }
        },
        required: ['appName', 'permission']
      } as any,
      handler(params?: { appName: string; permission: string; scope?: string }) {
        console.log('disable_app_permission', params);
        try {
          if (!params) {
            return enforceResponseLimit(JSON.stringify({ error: 'Invalid parameters' }));
          }

          if (platform() !== 'win32') {
            return enforceResponseLimit(JSON.stringify({ error: 'This function only works on Windows' }));
          }

          const scope = params.scope || 'user';
          const paths = PERMISSION_PATHS[params.permission];
          if (!paths) {
            return enforceResponseLimit(JSON.stringify({ error: 'Invalid permission type' }));
          }

          const path = paths[scope];
          if (!path) {
            return enforceResponseLimit(JSON.stringify({ error: 'Invalid scope. Use "user" or "system"' }));
          }

          const appPath = `${path}\\${params.appName}`;
          const command = `reg add "${appPath}" /v Value /t REG_SZ /d Deny /f`;
          execSync(command, { encoding: 'utf8' });

          return enforceResponseLimit(JSON.stringify({
            success: true,
            app: params.appName,
            permission: params.permission,
            scope: scope,
            message: `${params.permission} access disabled for app at ${scope} level`
          }));
        } catch (error: any) {
          return enforceResponseLimit(JSON.stringify({ error: error.message }));
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

  private markdownSearch(html: string): string {
    // Extract all anchor tags with href attributes
    const linkMatches = [
      ...Array.from(html.matchAll(/<a[^>]*href="([^"]*)"[^>]*>(.*?)<\/a>/gi)),
      ...Array.from(html.matchAll(/<a[^>]*href='([^']*)'[^>]*>(.*?)<\/a>/gi))
    ];

    const links: string[] = [];
    const seenUrls = new Set<string>();

    for (const match of linkMatches) {
      const url = match[1];
      let text = match[2];

      // Skip empty URLs or anchors
      if (!url || url === '#' || url.startsWith('javascript:')) {
        continue;
      }

      // Skip duplicates
      if (seenUrls.has(url)) {
        continue;
      }
      seenUrls.add(url);

      // Remove HTML tags from link text
      text = text.replace(/<\/?[^>]+(>|$)/g, '');

      // Decode HTML entities in text
      text = text.replace(/&nbsp;/g, ' ');
      text = text.replace(/&amp;/g, '&');
      text = text.replace(/&lt;/g, '<');
      text = text.replace(/&gt;/g, '>');
      text = text.replace(/&quot;/g, '"');
      text = text.replace(/&#39;/g, "'");
      text = text.replace(/&mdash;/g, '—');
      text = text.replace(/&ndash;/g, '–');
      text = text.replace(/&hellip;/g, '...');

      // Clean up whitespace in text
      text = text.replace(/\s+/g, ' ').trim();

      // Skip if text is empty after cleaning
      if (!text) {
        continue;
      }

      // Format as markdown link
      links.push(`[${text}](${url})`);
    }

    return links.join('\n');
  }

  private markdownContent(html: string): string {
    // Extract body content
    const bodyMatch = html.match(/<body[^>]*>((.|[\n\r])*)<\/body>/i);
    let content = bodyMatch ? bodyMatch[1] : html;

    // Remove scripts and styles
    content = content.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
    content = content.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '');
    content = content.replace(/<noscript\b[^<]*(?:(?!<\/noscript>)<[^<]*)*<\/noscript>/gi, '');

    // Convert headings (h1-h6)
    content = content.replace(/<h1[^>]*>(.*?)<\/h1>/gi, '\n# $1\n\n');
    content = content.replace(/<h2[^>]*>(.*?)<\/h2>/gi, '\n## $1\n\n');
    content = content.replace(/<h3[^>]*>(.*?)<\/h3>/gi, '\n### $1\n\n');
    content = content.replace(/<h4[^>]*>(.*?)<\/h4>/gi, '\n#### $1\n\n');
    content = content.replace(/<h5[^>]*>(.*?)<\/h5>/gi, '\n##### $1\n\n');
    content = content.replace(/<h6[^>]*>(.*?)<\/h6>/gi, '\n###### $1\n\n');

    // Extract text from links (keep text, discard URL)
    content = content.replace(/<a[^>]*>(.*?)<\/a>/gi, '$1');

    // Remove images
    content = content.replace(/<img[^>]*>/gi, '');

    // Convert bold/strong
    content = content.replace(/<(strong|b)[^>]*>(.*?)<\/(strong|b)>/gi, '**$2**');

    // Convert italic/em
    content = content.replace(/<(em|i)[^>]*>(.*?)<\/(em|i)>/gi, '*$2*');

    // Convert blockquotes
    content = content.replace(/<blockquote[^>]*>(.*?)<\/blockquote>/gis, (match, quote) => {
      const lines = quote.trim().split('\n');
      return '\n' + lines.map((line: string) => '> ' + line.trim()).join('\n') + '\n\n';
    });

    // Convert unordered lists
    content = content.replace(/<ul[^>]*>(.*?)<\/ul>/gis, (match, list) => {
      const items = list.match(/<li[^>]*>(.*?)<\/li>/gis) || [];
      return '\n' + items.map((item: string) => {
        const text = item.replace(/<\/?li[^>]*>/gi, '').trim();
        return '- ' + text;
      }).join('\n') + '\n\n';
    });

    // Convert ordered lists
    content = content.replace(/<ol[^>]*>(.*?)<\/ol>/gis, (match, list) => {
      const items = list.match(/<li[^>]*>(.*?)<\/li>/gis) || [];
      return '\n' + items.map((item: string, index: number) => {
        const text = item.replace(/<\/?li[^>]*>/gi, '').trim();
        return `${index + 1}. ${text}`;
      }).join('\n') + '\n\n';
    });

    // Convert horizontal rules
    content = content.replace(/<hr[^>]*>/gi, '\n---\n');

    // Convert paragraphs
    content = content.replace(/<p[^>]*>(.*?)<\/p>/gis, '\n$1\n\n');

    // Convert line breaks
    content = content.replace(/<br\s*\/?>/gi, '\n');

    // Convert divs to line breaks
    content = content.replace(/<div[^>]*>/gi, '\n');
    content = content.replace(/<\/div>/gi, '\n');

    // Decode HTML entities
    content = content.replace(/&nbsp;/g, ' ');
    content = content.replace(/&amp;/g, '&');
    content = content.replace(/&lt;/g, '<');
    content = content.replace(/&gt;/g, '>');
    content = content.replace(/&quot;/g, '"');
    content = content.replace(/&#39;/g, "'");
    content = content.replace(/&mdash;/g, '—');
    content = content.replace(/&ndash;/g, '–');
    content = content.replace(/&hellip;/g, '...');

    // Remove remaining HTML tags
    content = content.replace(/<\/?[^>]+(>|$)/g, '');

    // Clean up excessive whitespace
    content = content.replace(/\n{3,}/g, '\n\n');
    content = content.replace(/[ \t]+/g, ' ');
    content = content.trim();

    return content;
  }
}
