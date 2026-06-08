package com.example.rag.sync;

import com.example.rag.config.RagProperties;
import com.example.rag.service.RagService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 文档同步调度：
 *  - 启动后立即同步一次（受 rag.docs.sync-on-startup 控制）
 *  - 按 cron 表达式定时同步（默认每小时整点）
 *  - 通过 {@link #triggerSync(String)} 提供手动触发
 *
 * 同一时刻仅允许一个同步任务在跑（runOnce 互斥）。
 */
@Component
public class DocumentSyncScheduler {

    private static final Logger log = LoggerFactory.getLogger(DocumentSyncScheduler.class);

    private final RagProperties props;
    private final RagService ragService;
    private final AtomicBoolean running = new AtomicBoolean(false);

    public DocumentSyncScheduler(RagProperties props, RagService ragService) {
        this.props = props;
        this.ragService = ragService;
    }

    @EventListener(ApplicationReadyEvent.class)
    public void onAppReady() {
        if (props.getDocs().isSyncOnStartup()) {
            runSafely("startup");
        } else {
            log.info("Sync on startup disabled (rag.docs.sync-on-startup=false)");
        }
    }

    /** 默认每小时整点一次；可通过 rag.docs.sync-cron 修改。 */
    @Scheduled(cron = "${rag.docs.sync-cron}")
    public void scheduled() {
        runSafely("scheduled");
    }

    /**
     * 手动触发同步；用于 HTTP 接口等场景。
     * 已在跑则跳过（不抛错），返回 null。
     */
    public RagService.SyncResult triggerSync(String source) {
        if (!running.compareAndSet(false, true)) {
            log.info("Sync already in progress, skip manual trigger (source={})", source);
            return null;
        }
        try {
            log.info("Sync triggered by {}", source);
            RagService.SyncResult r = ragService.syncAll();
            log.info("Sync finished (source={}): {}", source, r);
            return r;
        } catch (Exception e) {
            log.error("Sync failed (source={}): {}", source, e.getMessage(), e);
            throw new IllegalStateException("Sync failed: " + e.getMessage(), e);
        } finally {
            running.set(false);
        }
    }

    private void runSafely(String source) {
        try {
            triggerSync(source);
        } catch (Exception e) {
            // 调度线程吞掉异常，避免下一次调度被取消
            log.error("Scheduled sync crashed (source={})", source, e);
        }
    }
}
