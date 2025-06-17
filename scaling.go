package main

import (
	"log"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

type ScalingMetrics struct {
	GPUUtilization float64
	VRAMUsed       float64
	Requests       int
}

var (
	scalingLock    sync.Mutex
	currentWorkers = make(map[string]int) // model -> worker count
	requestCounts  = make(map[string]int) // model -> request count
	nextPort       = 11434                // Starting port number
)

// StartScalingManager runs the auto-scaling loop
func StartScalingManager() {
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()

		for range ticker.C {
			scaleModelsBasedOnDemand()
		}
	}()
}

// TrackRequest increments counters for scaling decisions
func TrackRequest(model string) {
	scalingLock.Lock()
	defer scalingLock.Unlock()
	requestCounts[model]++
}

func scaleModelsBasedOnDemand() {
	scalingLock.Lock()
	defer scalingLock.Unlock()

	metrics := getSystemMetrics()
	models := getActiveModels()

	for _, model := range models {
		requests := requestCounts[model]
		workers := currentWorkers[model]
		desiredWorkers := calculateDesiredWorkers(model, requests, metrics)

		if desiredWorkers > workers {
			scaleUpModel(model, desiredWorkers-workers)
		} else if desiredWorkers < workers {
			scaleDownModel(model, workers-desiredWorkers)
		}

		requestCounts[model] = 0 // Reset counter after scaling decision
	}
}

func calculateDesiredWorkers(model string, requests int, metrics ScalingMetrics) int {
	base := 1 // Minimum workers

	// Scale based on GPU utilization
	if metrics.GPUUtilization > 70 {
		base++
	}

	// Scale based on request load
	if requests > 10 {
		base += requests / 5
	}

	// Don't exceed VRAM capacity
	maxByVRAM := int((metrics.VRAMUsed * 0.9) / 4000) // Assuming 4GB per model
	if base > maxByVRAM {
		return maxByVRAM
	}

	return base
}

func scaleUpModel(model string, count int) {
	for i := 0; i < count; i++ {
		port := findAvailablePort()
		if err := startOllamaContainer(model, port); err != nil {
			log.Printf("Failed to scale up %s: %v", model, err)
			continue
		}
		currentWorkers[model]++
		log.Printf("Scaled up %s to %d workers", model, currentWorkers[model])
	}
}

func scaleDownModel(model string, count int) {
	ports := getModelPorts(model)
	for i := 0; i < count && len(ports) > 1; i++ {
		port := ports[len(ports)-1]
		if err := stopOllamaContainer(model, port); err != nil {
			log.Printf("Failed to scale down %s: %v", model, err)
			continue
		}
		currentWorkers[model]--
		log.Printf("Scaled down %s to %d workers", model, currentWorkers[model])
	}
}

func stopOllamaContainer(model, port string) error {
	activeModels.Delete(model)
	return exec.Command("docker", "stop", "ollama-"+model+"-"+port).Run()
}

func getSystemMetrics() ScalingMetrics {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=utilization.gpu,memory.used",
		"--format=csv,noheader,nounits").Output()

	if err != nil {
		log.Printf("Failed to get GPU metrics: %v", err)
		return ScalingMetrics{}
	}

	parts := strings.Split(strings.TrimSpace(string(out)), ",")
	gpuUtil, _ := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	vramUsed, _ := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)

	return ScalingMetrics{
		GPUUtilization: gpuUtil,
		VRAMUsed:       vramUsed,
	}
}

func getModelPorts(model string) []string {
	var ports []string
	activeModels.Range(func(key, value interface{}) bool {
		if key.(string) == model {
			ports = append(ports, value.(string))
		}
		return true
	})
	return ports
}

// findAvailablePort returns the next available port number
func findAvailablePort() string {
	scalingLock.Lock()
	defer scalingLock.Unlock()

	port := nextPort
	nextPort++ // Increment for next call
	return strconv.Itoa(port)
}

// getActiveModels returns list of active models
func getActiveModels() []string {
	var models []string
	activeModels.Range(func(key, _ interface{}) bool {
		models = append(models, key.(string))
		return true
	})
	return models
}
