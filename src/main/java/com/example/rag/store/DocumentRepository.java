package com.example.rag.store;

import com.example.rag.config.RagProperties;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Stream;

/**
 * 原始文档仓库：负责上传文件落盘、扫描目录下文件、读取与删除。
 *
 * 路径解析：
 *  - classpath: 前缀：只读模式（扫描 / 读取）。上传 / 删除将抛异常。
 *  - 文件系统路径：读写都支持；启动时自动建目录。
 */
@Component
public class DocumentRepository {

    private static final Logger log = LoggerFactory.getLogger(DocumentRepository.class);

    private static final String CLASSPATH_PREFIX = "classpath:";
    private static final Set<String> SUPPORTED_EXTS = Set.of(".txt", ".csv", ".md");

    private final RagProperties props;
    private Path fsRoot;
    private String classpathRoot;

    public DocumentRepository(RagProperties props) {
        this.props = props;
    }

    @PostConstruct
    public void init() throws IOException {
        String raw = props.getDocs().getPath();
        if (raw == null || raw.isBlank()) {
            throw new IllegalStateException("rag.docs.path 未配置");
        }
        String trimmed = raw.trim();
        if (trimmed.startsWith(CLASSPATH_PREFIX)) {
            this.classpathRoot = trimmed.substring(CLASSPATH_PREFIX.length()).replaceAll("/+$", "");
            this.fsRoot = null;
            log.info("DocumentRepository in classpath mode: {}", classpathRoot);
        } else {
            this.fsRoot = Paths.get(trimmed).toAbsolutePath().normalize();
            this.classpathRoot = null;
            Files.createDirectories(fsRoot);
            log.info("DocumentRepository in filesystem mode: {}", fsRoot);
        }
    }

    public boolean isClasspath() { return fsRoot == null; }

    public Path getFsRoot() { return fsRoot; }

    public String getClasspathRoot() { return classpathRoot; }

    /**
     * 把上传文件落到 rag.docs.path 下，文件名 = `{uuid}_{originalName}`，避免同名覆盖。
     */
    public StoredFile save(MultipartFile file) throws IOException {
        requireFilesystem("upload");
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("file is required and must not be empty");
        }
        String original = file.getOriginalFilename();
        if (original == null || original.isBlank()) {
            throw new IllegalArgumentException("file must have a valid filename");
        }
        String safe = Paths.get(original).getFileName().toString();
        if (!isSupported(safe)) {
            throw new IllegalArgumentException(
                    "Unsupported file extension: " + safe + " (allowed: " + SUPPORTED_EXTS + ")");
        }
        long size = file.getSize();
        long max = props.getDocs().getMaxFileSize();
        if (size > max) {
            throw new IllegalArgumentException(
                    "file size " + size + " exceeds limit " + max);
        }
        String storedName = UUID.randomUUID() + "_" + safe;
        Path target = fsRoot.resolve(storedName);
        try (InputStream in = file.getInputStream()) {
            Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
        }
        long saved = Files.size(target);
        log.info("Saved upload: original='{}' stored='{}' size={}", original, storedName, saved);
        return new StoredFile(storedName, target, saved);
    }

    /** 扫描目录下所有支持的文件。 */
    public List<ScannedFile> scan() throws IOException {
        if (fsRoot != null) {
            return scanFilesystem();
        }
        return scanClasspath();
    }

    private List<ScannedFile> scanFilesystem() throws IOException {
        if (!Files.isDirectory(fsRoot)) return List.of();
        List<ScannedFile> out = new ArrayList<>();
        try (Stream<Path> stream = Files.list(fsRoot)) {
            Iterable<Path> it = stream::iterator;
            for (Path p : it) {
                if (!Files.isRegularFile(p)) continue;
                String name = p.getFileName().toString();
                if (!isSupported(name)) continue;
                out.add(new ScannedFile(name, p, null, Files.size(p)));
            }
        }
        return out;
    }

    private List<ScannedFile> scanClasspath() throws IOException {
        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        Resource[] resources;
        try {
            resources = resolver.getResources(
                    ResourceLoader.CLASSPATH_URL_PREFIX + classpathRoot + "/**");
        } catch (IOException e) {
            throw new IllegalStateException("classpath 下未找到文档目录: " + classpathRoot, e);
        }
        List<ScannedFile> out = new ArrayList<>();
        for (Resource r : resources) {
            if (!r.isReadable() || r.getFilename() == null) continue;
            String name = r.getFilename();
            if (name.endsWith("/")) continue;
            if (!isSupported(name)) continue;
            long size;
            try {
                size = r.contentLength();
            } catch (IOException e) {
                size = -1L;
            }
            out.add(new ScannedFile(name, null, r, size));
        }
        return out;
    }

    /** 按 filename 读取字节内容。 */
    public byte[] read(ScannedFile sf) throws IOException {
        if (sf.path != null) {
            return Files.readAllBytes(sf.path);
        }
        if (sf.resource != null) {
            try (InputStream in = sf.resource.getInputStream()) {
                return in.readAllBytes();
            }
        }
        throw new IllegalStateException("ScannedFile has neither path nor resource: " + sf.filename);
    }

    /** 删除文件系统中已落盘的文件。classpath 模式不允许删除。 */
    public boolean delete(String filename) throws IOException {
        requireFilesystem("delete");
        if (filename == null || filename.isBlank()) {
            throw new IllegalArgumentException("filename is required");
        }
        // 防路径穿越：只取 basename
        String safe = Paths.get(filename).getFileName().toString();
        return Files.deleteIfExists(fsRoot.resolve(safe));
    }

    private void requireFilesystem(String op) {
        if (fsRoot == null) {
            throw new IllegalStateException(
                    "rag.docs.path is classpath mode; operation '" + op + "' is not allowed. "
                            + "请把 rag.docs.path 改为文件系统路径以启用 " + op + "。");
        }
    }

    public static boolean isSupported(String filename) {
        if (filename == null) return false;
        String lower = filename.toLowerCase(Locale.ROOT);
        for (String ext : SUPPORTED_EXTS) {
            if (lower.endsWith(ext)) return true;
        }
        return false;
    }

    /** 上传后写入的文件信息。 */
    public record StoredFile(String storedName, Path path, long size) {}

    /**
     * 扫描得到的文件。path 与 resource 至少一个非空：
     *  - 文件系统模式：path 非空
     *  - classpath 模式：resource 非空
     */
    public record ScannedFile(String filename, Path path, Resource resource, long size) {
        public byte[] read() throws IOException {
            if (path != null) return Files.readAllBytes(path);
            try (InputStream in = resource.getInputStream()) {
                return in.readAllBytes();
            }
        }

        public boolean isClasspath() { return resource != null; }
    }
}
