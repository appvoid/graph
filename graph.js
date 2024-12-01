class graph {
  constructor(options = {}) {
    this.config = null;
    this.weights = null;
    this.debug = options.debug || false;
    this.ready = false;
  }

  static addMatrices(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a[0].length !== b[0].length) {
      throw new Error("Incompatible matrix dimensions for addition");
    }
    return a.map((row, i) =>
      row.map((val, j) => val + b[i][j])
    );
  }

  static matMul(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b)) {
      throw new Error("Invalid input matrices");
    }
    if (a[0].length !== b.length) {
      throw new Error("Incompatible matrix dimensions for multiplication");
    }
    const result = [];
    for (let i = 0; i < a.length; i++) {
      const row = [];
      for (let j = 0; j < b[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < a[i].length; k++) {
          sum += a[i][k] * b[k][j];
        }
        row.push(sum);
      }
      result.push(row);
    }
    return result;
  }

  static addBias(matrix, bias) {
    if (!Array.isArray(matrix) || !Array.isArray(bias)) {
      throw new Error("Invalid input for addBias");
    }
    if (matrix[0].length !== bias.length) {
      throw new Error("Incompatible dimensions for bias addition");
    }
    return matrix.map((row, i) =>
      row.map((val, j) => val + (bias[j] || 0))
    );
  }

  static softmax(logits) {
    if (!Array.isArray(logits)) {
      throw new Error("Invalid input for softmax");
    }
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
  }

  getEmbedding(tokens) {
    if (!this.weights || !this.weights['embedding.weight']) {
      throw new Error("Weights not loaded");
    }
    return tokens.map(token => this.weights['embedding.weight'][token]);
  }

  multiHeadAttention(x, layer_weights) {
    if (!layer_weights['self_attn.in_proj_weight'] ||
      !layer_weights['self_attn.in_proj_bias'] ||
      !layer_weights['self_attn.out_proj.weight'] ||
      !layer_weights['self_attn.out_proj.bias']) {
      throw new Error("Missing attention layer weights");
    }

    const projWeight = layer_weights['self_attn.in_proj_weight'];
    const projBias = layer_weights['self_attn.in_proj_bias'];
    const outProjWeight = layer_weights['self_attn.out_proj.weight'];
    const outProjBias = layer_weights['self_attn.out_proj.bias'];

    // Transpose projWeight to match dimensions for multiplication
    const projWeightT = graph.transpose(projWeight);

    // Multiply x with transposed projWeight
    const proj = graph.matMul(x, projWeightT);

    // Split proj into Q, K, V matrices
    const embed_size = this.config.embed_size;
    const q = proj.map(row => row.slice(0, embed_size));
    const k = proj.map(row => row.slice(embed_size, 2 * embed_size));
    const v = proj.map(row => row.slice(2 * embed_size, 3 * embed_size));

    // Split projBias into Q_bias, K_bias, V_bias
    const qBias = projBias.slice(0, embed_size);
    const kBias = projBias.slice(embed_size, 2 * embed_size);
    const vBias = projBias.slice(2 * embed_size, 3 * embed_size);

    // Add biases to Q, K, V
    const qWithBias = graph.addBias(q, qBias);
    const kWithBias = graph.addBias(k, kBias);
    const vWithBias = graph.addBias(v, vBias);

    const sqrtD = Math.sqrt(this.config.embed_size);
    const scores = graph.matMul(qWithBias, graph.transpose(kWithBias));
    const scaledScores = scores.map(row => row.map(x => x / sqrtD));
    const attnProbs = scaledScores.map(row => graph.softmax(row));

    const attnOutput = graph.matMul(attnProbs, vWithBias);
    const outProj = graph.matMul(attnOutput, outProjWeight);
    return graph.addBias(outProj, outProjBias);
  }

  transformerLayer(x, layer_weights) {
    const attnOutput = this.multiHeadAttention(x, layer_weights);
    const residual1 = graph.addMatrices(attnOutput, x);
    const ff1 = graph.matMul(residual1, graph.transpose(layer_weights['linear1.weight']));
    const ff1WithBias = graph.addBias(ff1, layer_weights['linear1.bias']);
    const ff1Activated = ff1WithBias.map(row => row.map(x => Math.max(0, x)));
    const ff2 = graph.matMul(ff1Activated, graph.transpose(layer_weights['linear2.weight']));
    const ff2WithBias = graph.addBias(ff2, layer_weights['linear2.bias']);
    const residual2 = graph.addMatrices(ff2WithBias, residual1);
    return residual2;
  }
  predict(inputSequence, max_length = this.config.context_length) {
    if (!this.ready) {
      throw new Error("Model not loaded. Call load() first.");
    }

    let x = this.getEmbedding(inputSequence);

    for (let layer_weights of this.weights['transformer.layers']) {
      x = this.transformerLayer(x, layer_weights);
    }

    const lastHidden = x[x.length - 1];
    const logits = graph.matMul([lastHidden], graph.transpose(this.weights['output.weight']));
    const withBias = graph.addBias(logits, this.weights['output.bias']);
    const probabilities = graph.softmax(withBias[0]);
    const nextToken = probabilities.indexOf(Math.max(...probabilities));

    return [...inputSequence, nextToken].pop();
  }

  load(callback) {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.style.display = 'none';
    document.body.appendChild(input);

    input.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const modelData = JSON.parse(e.target.result);
          if (!modelData.config || !modelData.weights) {
            throw new Error("Invalid model format");
          }
          this.config = modelData.config;
          this.weights = modelData.weights;
          // logs...
          this.ready = true;
          if (this.debug) {
            console.log("✅ Model loaded successfully!", this.config);
          }
          // Call the callback after loading is complete
          callback();
        } catch (error) {
          if (this.debug) {
            console.error("❌ Failed to load model:", error);
          }
        }
      };
      reader.readAsText(file);
    };
    input.click();
    document.body.removeChild(input);
  }

  static transpose(matrix) {
    return matrix[0].map((col, i) => matrix.map(row => row[i]));
  }
}

// // Usage example
// const model = new graph();

// model.load(() => {
//   console.log("Next token:", model.predict([1, 2, 3]));
// })
