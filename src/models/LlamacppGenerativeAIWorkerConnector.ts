import { rmSync } from 'fs';

import { EventEmitter } from 'events';

import type { Llama, LlamaEmbeddingContext, LlamaContext, ChatHistoryItem, LlamaChatSession, LlamaContextSequence } from 'node-llama-cpp';

import type { GenerativeAIEngineType, GenerativeAIModelOutputType, IGenerativeAIModel, IGenerativeAIWorkerConnector, IGenerativeAIWorkerOptions, GenerativeAIWorkerConnectorParameters, GenerativeAIWorkerConnectorResult, IGenerativeAIPromptWorkerConnectorParameters, IGenerativeAIWorkerConnectorPromptResult, GenerativeAIWorkerConnectorTypes, IPromptFunction } from '@crewdle/web-sdk-types';

import { ILlamacppGenerativeAIWorkerOptions } from './LlamacppGenerativeAIWorkerOptions';
import { ILlamacppGenerativeAIWorkerModel } from './LlamacppGenerativeAIWorkerModel';

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
  private maxTokens = 1024;

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
  private static models: Map<string, ILlamacppGenerativeAIWorkerModel> = new Map();

  /**
   * The context.
   * @ignore
   */
  private static context?: {
    modelId: string;
    instance: LlamaContext;
  }

  /**
   * The embedding context.
   * @ignore
   */
  private static embeddingContext?: {
    modelId: string;
    instance: LlamaEmbeddingContext;
  }

  /**
   * The constructor.
   * @param options The options.
   */
  constructor(
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
  }

  /**
   * Get the VRAM state.
   * @returns The total and available VRAM.
   */
  static async getVramState(): Promise<{ total: number, available: number }> {
    const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
    const vramState = await engine.getVramState();
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
    if (this.engine) {
      return this.engine;
    }

    const { getLlama } = await import('node-llama-cpp');
    this.engine = await getLlama();
    return this.engine;
  }

  /**
   * Get a model.
   * @param id The model ID.
   * @returns The model.
   * @ignore
   */
  private static getModel(id: string): ILlamacppGenerativeAIWorkerModel | undefined {
    return this.models.get(id);
  }

  /**
   * Set a model.
   * @param id The model ID.
   * @param model The model.
   * @ignore
   */
  private static setModel(id: string, model: ILlamacppGenerativeAIWorkerModel): void {
    this.models.set(id, model);
  }

  /**
   * Delete a model.
   * @param id The model ID.
   * @ignore
   */
  private static deleteModel(id: string): void {
    this.models.delete(id);
  }

  /**
   * Initialize the machine learning model.
   * @param workflowId The workflow ID.
   * @param models The models to initialize.
   */
  async initialize(workflowId: string, models: Map<string, IGenerativeAIModel>): Promise<void> {
    const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
    for (const [modelName, modelObj] of models) {
      if (modelObj.engineType !== 'llamacpp' || !modelObj.pathName) {
        continue;
      }
      let model = LlamacppGenerativeAIWorkerConnector.getModel(modelName);
      if (!model) {
        try {
          if (modelObj.outputType !== 'text' as GenerativeAIModelOutputType.Text) {
            const modelInstance = await engine.loadModel({
              modelPath: modelObj.pathName,
              useMlock: false,
            });
            model = {
              model: modelInstance,
              workflows: new Set(),
            }
          } else {
            const modelInstance = await engine.loadModel({
              modelPath: modelObj.pathName,
              useMlock: false,
              defaultContextFlashAttention: true,
            });
            model = {
              model: modelInstance,
              workflows: new Set(),
            }
          }

          model.workflows.add(workflowId);
          LlamacppGenerativeAIWorkerConnector.setModel(modelName, model);
          this.workflowId = workflowId;
        } catch (e) {
          console.error(e);
          rmSync(modelObj.pathName);
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
    for (const [id, model] of LlamacppGenerativeAIWorkerConnector.models) {
      model.workflows.delete(this.workflowId);
      if (model.workflows.size === 0) {
        if (LlamacppGenerativeAIWorkerConnector.context?.modelId === id) {
          await LlamacppGenerativeAIWorkerConnector.context.instance.dispose();
          LlamacppGenerativeAIWorkerConnector.context = undefined;
        }
        if (LlamacppGenerativeAIWorkerConnector.embeddingContext?.modelId === id) {
          await LlamacppGenerativeAIWorkerConnector.embeddingContext.instance.dispose();
          LlamacppGenerativeAIWorkerConnector.embeddingContext = undefined;
        }
        await model.model.dispose();
        LlamacppGenerativeAIWorkerConnector.deleteModel(id);
      } else {
        LlamacppGenerativeAIWorkerConnector.setModel(id, model);
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
    const model = LlamacppGenerativeAIWorkerConnector.getModel(options.model.id)?.model;
    if (!model) {
      throw new Error('Model not initialized');
    }

    if (options.model.outputType === 'vector' as GenerativeAIModelOutputType.Vector) {
      if (!LlamacppGenerativeAIWorkerConnector.embeddingContext || LlamacppGenerativeAIWorkerConnector.embeddingContext.modelId !== options.model.id) {
        if (LlamacppGenerativeAIWorkerConnector.embeddingContext) {
          await LlamacppGenerativeAIWorkerConnector.embeddingContext.instance.dispose();
        }
        const instance = await model.createEmbeddingContext();
        LlamacppGenerativeAIWorkerConnector.embeddingContext = {
          modelId: options.model.id,
          instance,
        };
      }
      const vector = await this.getVector(LlamacppGenerativeAIWorkerConnector.embeddingContext.instance, parameters.prompt);
      return {
        type: 'prompt' as GenerativeAIWorkerConnectorTypes,
        output: vector,
      };
    }

    if (!LlamacppGenerativeAIWorkerConnector.context || LlamacppGenerativeAIWorkerConnector.context.modelId !== options.model.id) {
      if (LlamacppGenerativeAIWorkerConnector.context) {
        await LlamacppGenerativeAIWorkerConnector.context.instance.dispose();
      }
      const instance = await model.createContext({
        sequences: 3,
      });
      LlamacppGenerativeAIWorkerConnector.context = {
        modelId: options.model.id,
        instance,
      };
    }

    const { prompt, functions, grammar, maxTokens, temperature, instructions } = parameters;
    const sequence = LlamacppGenerativeAIWorkerConnector.context.instance.getSequence();
    const { LlamaChatSession } = await import('node-llama-cpp');
    const session = new LlamaChatSession({
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
      functions: functions ? await this.getFunctions(functions) : undefined,
      grammar: undefined,
    }
    if (grammar) {
      if (grammar === 'json' || grammar === 'json_arr') {
        promptOptions.functions = undefined;
        promptOptions.grammar = await (await LlamacppGenerativeAIWorkerConnector.getEngine()).getGrammarFor(grammar);
      } else if (grammar !== 'default') {
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
    const outputTokens = sequence.tokenMeter.usedOutputTokens - startingOutputTokens

    session.dispose();
    sequence.dispose();

    return {
      type: 'prompt' as GenerativeAIWorkerConnectorTypes,
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
  async *processJobStream(parameters: IGenerativeAIPromptWorkerConnectorParameters, options: IGenerativeAIWorkerOptions): AsyncGenerator<IGenerativeAIWorkerConnectorPromptResult> {
    const model = LlamacppGenerativeAIWorkerConnector.getModel(options.model.id)?.model;
    if (!model) {
      throw new Error('Model not initialized');
    }

    if (options.model.outputType === 'vector' as GenerativeAIModelOutputType.Vector) {
      throw new Error('Vector output type not supported for streaming');
    }

    if (!LlamacppGenerativeAIWorkerConnector.context || LlamacppGenerativeAIWorkerConnector.context.modelId !== options.model.id) {
      if (LlamacppGenerativeAIWorkerConnector.context) {
        await LlamacppGenerativeAIWorkerConnector.context.instance.dispose();
      }
      const instance = await model.createContext({
        sequences: 3,
      });
      LlamacppGenerativeAIWorkerConnector.context = {
        modelId: options.model.id,
        instance,
      };
    }

    const { prompt, functions, grammar, maxTokens, temperature, instructions } = parameters;
    const sequence = LlamacppGenerativeAIWorkerConnector.context.instance.getSequence();
    const { LlamaChatSession } = await import('node-llama-cpp');
    const session = new LlamaChatSession({
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
      functions: functions ? await this.getFunctions(functions) : undefined,
      grammar: undefined,
    }
    if (grammar) {
      if (grammar === 'json' || grammar === 'json_arr') {
        promptOptions.functions = undefined;
        promptOptions.grammar = await (await LlamacppGenerativeAIWorkerConnector.getEngine()).getGrammarFor(grammar);
      } else if (grammar !== 'default') {
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
    return text.trim().toLowerCase().replace(/[^a-z0-9\s]/g, '');
  }

  /**
   * Normalize a vector.
   * @param vector The vector to normalize.
   * @returns The normalized vector.
   * @ignore
   */
  private normalizeVector(vector: readonly number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    return vector.map((value) => value / norm);
  }

  /**
   * Get the functions object.
   * @param functions The functions to process
   * @returns A promise that resolves to the functions object
   */
  private async getFunctions(functions: Map<string, IPromptFunction>): Promise<{[key: string]: any}> {
    const { defineChatSessionFunction } = await import('node-llama-cpp');
    let functionsObj: {[key: string]: any} = {}

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
