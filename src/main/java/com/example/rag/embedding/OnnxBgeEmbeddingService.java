package com.example.rag.embedding;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.*;
import com.example.rag.config.RagProperties;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * 加载 HuggingFace BGE 系列 ONNX 编码器（model.onnx + tokenizer.json）并通过 ONNX Runtime 推理。
 *
 * 路径解析规则（由 {@link RagProperties#model} 配置）：
 *  - 以 {@code classpath:} 开头：把 classpath 下的模型目录完整复制到临时目录后加载，
 *    用于解决 Spring Boot fat jar 内资源无法直接喂给 ONNX Runtime 的问题。
 *  - 否则视为文件系统路径（绝对或相对路径均可）。
 *
 * 启动校验：path 非空、model.onnx 与 tokenizer.json 必须存在且为普通文件。
 */
@Service
public class OnnxBgeEmbeddingService {

    private static final Logger log = LoggerFactory.getLogger(OnnxBgeEmbeddingService.class);

    private static final String CLASSPATH_PREFIX = "classpath:";
    private static final String MODEL_FILE = "model.onnx";
    private static final String TOKENIZER_FILE = "tokenizer.json";

    private final RagProperties props;
    private OrtEnvironment env;
    private OrtSession session;
    private HuggingFaceTokenizer tokenizer;
    /** classpath 模式下生成的临时目录；关闭时清理。 */
    private Path tempModelDir;

    public OnnxBgeEmbeddingService(RagProperties props) {
        this.props = props;
    }

    @PostConstruct
    public void init() throws Exception {
        Path modelDir = resolveModelDir();
        Path modelFile = modelDir.resolve(MODEL_FILE);
        Path tokenizerFile = modelDir.resolve(TOKENIZER_FILE);
        validateRequiredFiles(modelDir, modelFile, tokenizerFile);

        log.info("Loading ONNX model from {}", modelDir);
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(ortSessionOptLevel());
        session = env.createSession(modelFile.toString(), opts);
        tokenizer = HuggingFaceTokenizer.newInstance(tokenizerFile);

        log.info("ONNX model loaded. inputs={} outputs={}",
                session.getInputInfo().keySet(), session.getOutputInfo().keySet());
    }

    @PreDestroy
    public void close() throws Exception {
        try {
            if (session != null) {
                session.close();
                session = null;
            }
        } finally {
            if (env != null) {
                env.close();
                env = null;
            }
            if (tempModelDir != null) {
                deleteRecursively(tempModelDir);
                tempModelDir = null;
            }
        }
    }

    /**
     * 解析模型目录。
     * classpath: 模式会把整个目录复制到临时目录并返回临时目录路径。
     */
    private Path resolveModelDir() throws IOException {
        String rawPath = props.getModel().getPath();
        if (rawPath == null || rawPath.isBlank()) {
            throw new IllegalStateException(
                    "rag.model.path 未配置，请在 application.yml 中设置 rag.model.path");
        }
        String trimmed = rawPath.trim();
        if (trimmed.startsWith(CLASSPATH_PREFIX)) {
            String resourcePath = trimmed.substring(CLASSPATH_PREFIX.length()).replaceAll("/+$", "");
            if (resourcePath.isEmpty()) {
                throw new IllegalStateException(
                        "classpath: 后必须指定资源路径，例如 classpath:model/bge-small-zh-v1.5");
            }
            return copyClasspathDirToTemp(resourcePath);
        }
        Path p = Path.of(trimmed);
        if (!Files.exists(p)) {
            throw new IllegalStateException("模型目录不存在: " + p);
        }
        if (!Files.isDirectory(p)) {
            throw new IllegalStateException("rag.model.path 指向的不是目录: " + p);
        }
        return p;
    }

    private Path copyClasspathDirToTemp(String resourcePath) throws IOException {
        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        Resource[] resources;
        try {
            resources = resolver.getResources(
                    ResourceLoader.CLASSPATH_URL_PREFIX + resourcePath + "/**");
        } catch (IOException e) {
            throw new IllegalStateException("classpath 下未找到模型目录: " + resourcePath, e);
        }
        if (resources == null || resources.length == 0) {
            throw new IllegalStateException("classpath 下未找到模型目录: " + resourcePath);
        }
        Path tmp = Files.createTempDirectory("rag-model-");
        boolean copied = false;
        for (Resource r : resources) {
            if (!r.isReadable() || r.getFilename() == null) continue;
            String name = r.getFilename();
            // 跳过目录型条目
            if (name.endsWith("/")) continue;
            Path target = tmp.resolve(name);
            try (InputStream in = r.getInputStream()) {
                Files.copy(in, target, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            copied = true;
        }
        if (!copied) {
            deleteRecursively(tmp);
            throw new IllegalStateException("classpath 模型目录为空: " + resourcePath);
        }
        this.tempModelDir = tmp;
        log.info("Copied classpath model '{}' to temp dir {}", resourcePath, tmp);
        return tmp;
    }

    private void validateRequiredFiles(Path modelDir, Path modelFile, Path tokenizerFile) {
        if (!Files.isRegularFile(modelFile)) {
            throw new IllegalStateException("缺少模型文件: " + modelFile
                    + " (expected: " + modelDir + "/" + MODEL_FILE + ")");
        }
        if (!Files.isRegularFile(tokenizerFile)) {
            throw new IllegalStateException("缺少 tokenizer 文件: " + tokenizerFile
                    + " (expected: " + modelDir + "/" + TOKENIZER_FILE + ")");
        }
    }

    private static void deleteRecursively(Path dir) {
        if (dir == null || !Files.exists(dir)) return;
        try (Stream<Path> stream = Files.walk(dir)) {
            stream.sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
                try {
                    Files.deleteIfExists(p);
                } catch (IOException ignored) {
                }
            });
        } catch (IOException ignored) {
        }
    }

    /** 抽成方法避免在 init() 中直接 new 内部类造成 IDE 误报。 */
    private static OrtSession.SessionOptions.OptLevel ortSessionOptLevel() {
        return OrtSession.SessionOptions.OptLevel.ALL_OPT;
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
            int hiddenSize;
            int outSeqLen;
            float[] flat;
            float[][][] hidden;
            if (raw instanceof float[][][]) {
                hidden = (float[][][]) raw;
                outSeqLen = hidden[0].length;
                hiddenSize = hidden[0][0].length;
                flat = null;
            } else if (raw instanceof float[][]) {
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
                outSeqLen = seqLen;
                if (outSeqLen <= 0) {
                    hiddenSize = 0;
                } else {
                    hiddenSize = flat.length / outSeqLen;
                }
                if (hiddenSize <= 0 && flat.length > 0) {
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

            double norm = 0;
            for (float v : sum) norm += v * v;
            norm = Math.sqrt(norm);
            if (norm > 0) for (int i = 0; i < sum.length; i++) sum[i] = (float) (sum[i] / norm);

            return sum;
        }
    }

    public int dimension() throws Exception {
        return embed("probe").length;
    }

    /** 测试辅助。 */
    List<String> tokenizeForDebug(String text) {
        return tokenizer.tokenize(text);
    }
}
