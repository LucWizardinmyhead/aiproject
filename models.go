package main

import (
	"encoding/json"
	"io"
	"net/http"
	"os/exec"
	"strconv"
	"sync"
)

var (
	activeModels sync.Map // Concurrent-safe map for model tracking
	basePort     = 11434  // Starting port for Ollama containers
)

type GenerationRequest struct {
	Model   string `json:"model"`
	Prompt  string `json:"prompt"`
	Context []int  `json:"context,omitempty"`
}

func generateHandler(w http.ResponseWriter, r *http.Request) {
	var req GenerationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondWithError(w, http.StatusBadRequest, "Invalid request format")
		return
	}

	TrackRequest(req.Model)
	port, err := ensureModelRunning(req.Model)
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, "Failed to start model: "+err.Error())
		return
	}

	forwardRequestToOllama(w, r, port)
}

func listModelsHandler(w http.ResponseWriter, r *http.Request) {
	models := getActiveModels()
	respondWithJSON(w, http.StatusOK, map[string]interface{}{
		"models": models,
		"count":  len(models),
	})
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	respondWithJSON(w, http.StatusOK, map[string]string{"status": "healthy"})
}

// Helper functions
func ensureModelRunning(model string) (string, error) {
	if port, exists := activeModels.Load(model); exists {
		return port.(string), nil
	}

	port := strconv.Itoa(basePort + countActiveModels())
	if err := startOllamaContainer(model, port); err != nil {
		return "", err
	}
	return port, nil
}

func startOllamaContainer(model, port string) error {
	containerName := "ollama-" + model + "-" + port

	// Start container
	if err := exec.Command("docker", "run", "--rm", "-d",
		"-p", port+":11434",
		"--name", containerName,
		"ollama/ollama",
	).Run(); err != nil {
		return err
	}

	// Pull model (non-blocking)
	go func() {
		exec.Command("docker", "exec", containerName, "ollama", "pull", model).Run()
	}()

	activeModels.Store(model, port)
	return nil
}

func forwardRequestToOllama(w http.ResponseWriter, r *http.Request, port string) {
	resp, err := http.Post(
		"http://localhost:"+port+"/api/generate",
		"application/json",
		r.Body,
	)
	if err != nil {
		respondWithError(w, http.StatusServiceUnavailable, "Model service unavailable")
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "application/json")
	io.Copy(w, resp.Body)
}

func getActiveModels() []string {
	var models []string
	activeModels.Range(func(key, _ interface{}) bool {
		models = append(models, key.(string))
		return true
	})
	return models
}

func countActiveModels() int {
	count := 0
	activeModels.Range(func(_, _ interface{}) bool {
		count++
		return true
	})
	return count
}

func respondWithJSON(w http.ResponseWriter, code int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(payload)
}

func respondWithError(w http.ResponseWriter, code int, message string) {
	respondWithJSON(w, code, map[string]string{"error": message})
}
