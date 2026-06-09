# ============================================================================
# spring-rag Docker 镜像
# 构建前置(宿主机执行): mvn -B -DskipTests package
#   产物: target/spring-rag.tar.gz  (由 maven-assembly-plugin 产出)
#         /spring-rag/bin/app(.bat)
#         /spring-rag/conf/         application.yml 等
#         /spring-rag/lib/*.jar     全部依赖 + 应用 jar(model.onnx 内嵌其中)
# ============================================================================

# ---------- 运行时:仅 JRE ----------
# 选 JRE 而非 JDK:JDK 多带 ~150MB,运行时不需要;ONNX Runtime 1.18.0 在 JRE 下完全工作
# 不要用 alpine:musl + ONNX Runtime 的 native .so 会触发 UnsatisfiedLinkError
FROM eclipse-temurin:17-jre

# 安装 curl 仅给 HEALTHCHECK 用,装完即清,不污染镜像
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# 非 root 用户(安全基线;容器逃逸风险)
RUN groupadd -r rag && useradd -r -g rag -d /app -s /sbin/nologin rag

# 解包 appassembler 产物
ADD target/spring-rag.tar.gz /app
RUN chmod +x /app/bin/* && chown -R rag:rag /app

# ----------------------------------------------------------------------------
# ONNX Runtime 关键环境变量
# ----------------------------------------------------------------------------
# LANG/LC_ALL=C.UTF-8
#   DJL HuggingFaceTokenizer 解析中文必须 UTF-8,否则 OOV / 乱码
# -XX:+UseG1GC
#   低延迟 GC,适配 Web 服务的请求响应模型
# -XX:MaxRAMPercentage=75.0
#   让 JVM 跟随容器内存(cgroup limits)自动伸缩 -Xmx,无需写死
# -XX:+ExitOnOutOfMemoryError
#   OOM 时快速失败,Docker 能及时重启而不是僵死
# -Dfile.encoding=UTF-8
#   Spring 内部 + Jackson 序列化兜底
# -Djava.io.tmpdir=/tmp
#   OnnxBgeEmbeddingService 在 classpath: 模式下把模型复制到
#   Files.createTempDirectory("rag-model-"),Linux 默认就是 /tmp,显式更稳
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    JAVA_OPTS="-XX:+UseG1GC -XX:MaxRAMPercentage=75.0 -XX:+ExitOnOutOfMemoryError -Dfile.encoding=UTF-8 -Djava.io.tmpdir=/tmp"

# 业务数据目录(运行时挂卷,例如: -v /opt/docs:/app/docs)
RUN mkdir -p /app/docs && chown -R rag:rag /app/docs

WORKDIR /app
EXPOSE 8080

# 健康检查:走 RAG 主链路,比 Spring 默认 /actuator/health 更贴业务
# start-period=60s 留足 ONNX 模型首次加载 + tokenizer 初始化的窗口(约 5-10s)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/api/rag/stats || exit 1

USER rag

# 绕过 JSW wrapper 直接 java 起:
#   1. JAVA_OPTS 真正生效(appassembler 的 JSW 包装不读环境变量)
#   2. Docker stop 能优雅关闭(JSW console 模式信号处理不友好)
#   3. classpath: conf(放 yml)+ lib/*(deps + app jar,内含 model.onnx)
CMD ["sh", "-c", "exec java $JAVA_OPTS -cp 'conf:lib/*' com.example.rag.RagApplication"]
