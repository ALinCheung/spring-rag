package com.example.rag.embedding;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.*;
import com.example.rag.config.RagProperties;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Loads a HuggingFace BGE-style encoder that has been exported to ONNX
 * (model.onnx + tokenizer.json in the same folder) and runs it via ONNX Runtime.
 *
 * Expected ONNX I/O for sentence-transformers (bge-small-zh-v1.5):
 *   inputs  : input_ids (int64 [B,T]), attention_mask (int64 [B,T])
 *   outputs : last_hidden_state (float32 [B,T,H])   -> we mean-pool ourselves
 *
 * The mean pooling mirrors the 1_Pooling/config.json used by sentence-transformers
 * (mean pooling, attention-mask aware). L2-normalisation is applied so cosine
 * similarity equals dot-product — handy for nearest-neighbour search.
 */
@Service
public class OnnxBgeEmbeddingService {

    private static final Logger log = LoggerFactory.getLogger(OnnxBgeEmbeddingService.class);

    private final RagProperties props;
    private OrtEnvironment env;
    private OrtSession session;
    private HuggingFaceTokenizer tokenizer;

    public OnnxBgeEmbeddingService(RagProperties props) {
        this.props = props;
    }

    @PostConstruct
    public void init() throws Exception {
        String modelDir = props.getModel().getPath();
        log.info("Loading ONNX model from {}", modelDir);

        Path modelFile = Path.of(modelDir, "model.onnx");
        Path tokenizerFile = Path.of(modelDir, "tokenizer.json");

        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        session = env.createSession(modelFile.toString(), opts);
        tokenizer = HuggingFaceTokenizer.newInstance(tokenizerFile);

        log.info("ONNX model loaded. inputs={} outputs={}",
                session.getInputInfo().keySet(), session.getOutputInfo().keySet());
    }

    @PreDestroy
    public void close() throws Exception {
        if (session != null) session.close();
        if (env != null) env.close();
    }

    public float[] embed(String text) throws Exception {
        if (text == null || text.isBlank()) {
            return new float[0];
        }
        Encoding encoding = tokenizer.encode(text);
        long[] ids = encoding.getIds();
        long[] mask = encoding.getAttentionMask();
        int seqLen = ids.length;

        long[][] inputIds = new long[1][seqLen];
        long[][] attnMask = new long[1][seqLen];
        long[][] typeIds = new long[1][seqLen];
        for (int i = 0; i < seqLen; i++) {
            inputIds[0][i] = ids[i];
            attnMask[0][i] = mask[i];
            typeIds[0][i] = 0L;
        }

        Map<String, OnnxTensor> feeds = new HashMap<>();
        feeds.put("input_ids", OnnxTensor.createTensor(env, inputIds));
        feeds.put("attention_mask", OnnxTensor.createTensor(env, attnMask));
        feeds.put("token_type_ids", OnnxTensor.createTensor(env, typeIds));

        try (OrtSession.Result result = session.run(feeds)) {
            OnnxValue lastHidden = result.get(0);
            Object raw = lastHidden.getValue();
            // ORT-Java returns the last_hidden_state tensor as a flat float[]
            // laid out in row-major order [B, T, H]. We reshape to [B][T][H].
            int hiddenSize = 0;
            int outSeqLen = 0;
            float[] flat;
            float[][][] hidden;
            if (raw instanceof float[][][]) {
                hidden = (float[][][]) raw;
                outSeqLen = hidden[0].length;
                hiddenSize = hidden[0][0].length;
                flat = null;
            } else if (raw instanceof float[][]) {
                // Some exports collapse the batch dim when batch=1 -> [T, H]
                float[][] twoD = (float[][]) raw;
                outSeqLen = twoD.length;
                hiddenSize = twoD[0].length;
                flat = new float[outSeqLen * hiddenSize];
                for (int t = 0; t < outSeqLen; t++) {
                    System.arraycopy(twoD[t], 0, flat, t * hiddenSize, hiddenSize);
                }
                hidden = null;
            } else {
                flat = (float[]) raw;
                // We need T and H to interpret the flat buffer. Probe the
                // session output metadata once via session.getOutputInfo()
                // (already done in init()), but here we keep things local:
                // assume the export uses dynamic axes so T == input seqLen,
                // and H == flat.length / T.
                outSeqLen = seqLen;
                if (outSeqLen <= 0) {
                    hiddenSize = 0;
                } else {
                    hiddenSize = flat.length / outSeqLen;
                }
                if (hiddenSize <= 0 && flat.length > 0) {
                    // Last-resort: try 384 (bge-small) then 768 (bge-base)
                    for (int guess : new int[]{384, 512, 768, 1024}) {
                        if (flat.length % guess == 0) {
                            hiddenSize = guess;
                            outSeqLen = flat.length / guess;
                            break;
                        }
                    }
                }
                hidden = null;
            }

            // Attention-mask aware mean pooling
            float[] sum = new float[hiddenSize];
            float weightSum = 0f;
            for (int t = 0; t < outSeqLen; t++) {
                float w = (float) mask[t];
                weightSum += w;
                for (int h = 0; h < hiddenSize; h++) {
                    float v;
                    if (hidden != null) {
                        v = hidden[0][t][h];
                    } else if (flat != null) {
                        v = flat[t * hiddenSize + h];
                    } else {
                        v = 0f;
                    }
                    sum[h] += v * w;
                }
            }
            if (weightSum == 0f) weightSum = 1f;
            for (int h = 0; h < hiddenSize; h++) sum[h] /= weightSum;

            // L2 normalise
            double norm = 0;
            for (float v : sum) norm += v * v;
            norm = Math.sqrt(norm);
            if (norm > 0) for (int i = 0; i < sum.length; i++) sum[i] = (float) (sum[i] / norm);

            return sum;
        }
    }

    public int dimension() throws Exception {
        // Derive from a probe embedding
        return embed("probe").length;
    }

    /** Test helper. */
    List<String> tokenizeForDebug(String text) {
        return tokenizer.tokenize(text);
    }
}
