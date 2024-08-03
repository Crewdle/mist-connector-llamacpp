import { EventEmitter } from 'events';

import type { Llama, LlamaModel, LlamaEmbeddingContext, LlamaContext } from 'node-llama-cpp';

import type { GenerativeAIModelOutputType, IGenerativeAIWorkerConnector, IGenerativeAIWorkerOptions, IJobParametersAI, IJobResultAI } from '@crewdle/web-sdk-types';

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
  async initialize(workflowId: string, models: Map<string, string>): Promise<void> {
    const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
    for (const [modelName, modelPath] of models) {
      let model = LlamacppGenerativeAIWorkerConnector.getModel(modelName);
      if (!model) {
        const modelInstance = await engine.loadModel({
          modelPath,
        });
        model = {
          model: modelInstance,
          workflows: new Set(),
        }
      }
      model.workflows.add(workflowId);
      LlamacppGenerativeAIWorkerConnector.setModel(modelName, model);
      this.workflowId = workflowId;
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

  /**
   * Process a job.
   * @param parameters The job parameters.
   * @returns A promise that resolves with the job result.
   */
  async processJob(parameters: IJobParametersAI, options: IGenerativeAIWorkerOptions): Promise<IJobResultAI> {
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
        output: vector,
      };
    }
    
    if (!LlamacppGenerativeAIWorkerConnector.context || LlamacppGenerativeAIWorkerConnector.context.modelId !== options.model.id) {
      if (LlamacppGenerativeAIWorkerConnector.context) {
        await LlamacppGenerativeAIWorkerConnector.context.instance.dispose();
      }
      const instance = await model.createContext();
      LlamacppGenerativeAIWorkerConnector.context = {
        modelId: options.model.id,
        instance,
      };
    }
    const { LlamaChatSession } = await import('node-llama-cpp');
    const session = new LlamaChatSession({
      contextSequence: LlamacppGenerativeAIWorkerConnector.context.instance.getSequence(),
    });

    const prompt = await this.getPrompt(parameters, model);

    const output = await session.prompt(prompt, {
      maxTokens: parameters.maxTokens ?? this.maxTokens,
      temperature: parameters.temperature ?? this.temperature,
    });

    const inputTokens = model.tokenize(prompt).length;
    const outputTokens = model.tokenize(output).length;
    
    session.dispose();

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
  async *processJobStream(parameters: IJobParametersAI, options: IGenerativeAIWorkerOptions): AsyncGenerator<IJobResultAI> {
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
      const instance = await model.createContext();
      LlamacppGenerativeAIWorkerConnector.context = {
        modelId: options.model.id,
        instance,
      };
    }
    const { LlamaChatSession } = await import('node-llama-cpp');
    const session = new LlamaChatSession({
      contextSequence: LlamacppGenerativeAIWorkerConnector.context.instance.getSequence(),
    });

    const prompt = await this.getPrompt(parameters, model);

    const inputTokens = model.tokenize(prompt).length;
    let outputTokens = 0;

    const textEmitter = new EventEmitter();

    session.prompt(prompt, {
      maxTokens: parameters.maxTokens ?? this.maxTokens,
      temperature: parameters.temperature ?? this.temperature,
      onTextChunk: (text) => {
        textEmitter.emit('text', text);
      },
    }).then(() => {
      textEmitter.emit('text', undefined);
    }).catch(e => {});

    while (true) {
      const text = await new Promise<string>((resolve) => textEmitter.once('text', resolve));

      if (text === undefined) {
        break;
      }

      outputTokens += model.tokenize(text).length;
      yield {
        output: text,
        inputTokens,
        outputTokens,
      };
    }

    session.dispose();
  }

  private async getPrompt(parameters: IJobParametersAI, model: LlamaModel): Promise<string> {
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
  private normalizeVector(vector: number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    return vector.map((value) => value / norm);
  }
}
