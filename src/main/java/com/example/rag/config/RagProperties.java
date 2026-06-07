package com.example.rag.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "rag")
public class RagProperties {

    private Model model = new Model();
    private Chunk chunk = new Chunk();

    public Model getModel() { return model; }
    public void setModel(Model model) { this.model = model; }
    public Chunk getChunk() { return chunk; }
    public void setChunk(Chunk chunk) { this.chunk = chunk; }

    public static class Model {
        private String path;
        public String getPath() { return path; }
        public void setPath(String path) { this.path = path; }
    }

    public static class Chunk {
        private int size = 200;
        private int overlap = 30;
        public int getSize() { return size; }
        public void setSize(int size) { this.size = size; }
        public int getOverlap() { return overlap; }
        public void setOverlap(int overlap) { this.overlap = overlap; }
    }
}
