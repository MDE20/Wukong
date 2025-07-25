package data

import (
	"../wukong"

	"fmt"
)

func Get(predName string) (wukong.PreprocessConfig, wukong.ModelConfig) {
	if predName == "shibuya" || predName == "shibuya-crosswalk" {
		return Shibuya(predName)
	} else if predName == "warsaw" || predName == "warsaw-brake" {
		return Warsaw(predName)
	} else if predName == "uav" {
		return UAV(predName)
	}
	panic(fmt.Errorf("unknown predicate %s", predName))
}

func GetExec(predName string) (detectionPath string, framePath string) {
	if predName == "shibuya" || predName == "shibuya-crosswalk" {
		detectionPath = "data/shibuya/json/2-detections.json"
		framePath = "data/shibuya/frames/2/"
	} else if predName == "warsaw" || predName == "warsaw-brake" {
		detectionPath = "data/warsaw/json/2-detections.json"
		framePath = "data/warsaw/frames/2/"
	} else if predName == "uav" {
		detectionPath = "data/uav/json/0011-detections.json"
		framePath = "data/uav/frames/0011/"
	} else {
		panic(fmt.Errorf("unknown predicate %s", predName))
	}
	return
}

func Shibuya(predName string) (wukong.PreprocessConfig, wukong.ModelConfig) {
	var segments []wukong.Segment
	for _, i := range []int{0, 1, 3, 4, 5} {
		segments = append(segments, wukong.Segment{
			FramePath: fmt.Sprintf("data/shibuya/frames/%d/", i),
			TrackPath: fmt.Sprintf("data/shibuya/json/%d-baseline.json", i),
		})
	}

	ppCfg := wukong.PreprocessConfig{
		TrainSegments: segments[0:2],
		ValSegments:   segments[2:],
		Predicate:     predName,
		FrameScale:    2,
	}
	var modelCfg wukong.ModelConfig
	for freq := 32; freq >= 1; freq /= 2 {
		modelCfg.GNN = append(modelCfg.GNN, wukong.GNNModel{
			Freq:      freq,
			ModelPath: "logs/shibuya/gnn/model",
		})
		modelCfg.Filters = append(modelCfg.Filters, wukong.FilterModel{
			Name: "rnn",
			Freq: freq,
			Cfg: map[string]string{
				"model_path": fmt.Sprintf("logs/%s/%d/filter-rnn/model", predName, freq),
			},
		})
		modelCfg.Refiners = append(modelCfg.Refiners, wukong.RefineModel{
			Name: "rnn",
			Freq: freq,
			Cfg: map[string]string{
				"model_path": fmt.Sprintf("logs/%s/%d/refine-rnn/model", predName, freq),
			},
		})
	}
	fmt.Printf("[Shibuya] modelCfg1: %+v\n", modelCfg)
	fmt.Printf("[Shibuya] modelCfg2: %+v\n", modelCfg)
	fmt.Printf("[Shibuya] modelCfg3: %+v\n", modelCfg)
	fmt.Printf("[Shibuya] modelCfg4: %+v\n", modelCfg)
	fmt.Printf("[Shibuya] modelCfg5: %+v\n", modelCfg)
	fmt.Printf("[Shibuya] modelCfg6: %+v\n", modelCfg)
	fmt.Printf("[Shibuya] modelCfg7: %+v\n", modelCfg)
	return ppCfg, modelCfg
}

func Warsaw(predName string) (wukong.PreprocessConfig, wukong.ModelConfig) {
	var segments []wukong.Segment
	for _, i := range []int{0, 1, 3, 4, 5} {
		segments = append(segments, wukong.Segment{
			FramePath: fmt.Sprintf("data/warsaw/frames/%d/", i),
			TrackPath: fmt.Sprintf("data/warsaw/json/%d-baseline.json", i),
		})
	}

	ppCfg := wukong.PreprocessConfig{
		TrainSegments: segments[0:2],
		ValSegments:   segments[2:],
		Predicate:     predName,
		FrameScale:    2,
	}
	var modelCfg wukong.ModelConfig
	for freq := 32; freq >= 1; freq /= 2 {
		modelCfg.GNN = append(modelCfg.GNN, wukong.GNNModel{
			Freq:      freq,
			ModelPath: "logs/warsaw/gnn/model",
		})
		modelCfg.Filters = append(modelCfg.Filters, wukong.FilterModel{
			Name: "rnn",
			Freq: freq,
			Cfg: map[string]string{
				"model_path": fmt.Sprintf("logs/%s/%d/filter-rnn/model", predName, freq),
			},
		})
		modelCfg.Refiners = append(modelCfg.Refiners, wukong.RefineModel{
			Name: "rnn",
			Freq: freq,
			Cfg: map[string]string{
				"model_path": fmt.Sprintf("logs/%s/%d/refine-rnn/model", predName, freq),
			},
		})
	}
	return ppCfg, modelCfg
}


func UAV(predName string) (wukong.PreprocessConfig, wukong.ModelConfig) {
	var segments []wukong.Segment
	for _, id := range []string{"0006", "0007", "0008", "0009"} {
		segments = append(segments, wukong.Segment{
			FramePath: fmt.Sprintf("data/uav/frames/%s/", id),
			TrackPath: fmt.Sprintf("data/uav/json/%s-baseline.json", id),
		})
	}

	ppCfg := wukong.PreprocessConfig{
		TrainSegments: segments[0:2],
		ValSegments:   segments[2:],
		Predicate:     predName,
		FrameScale:    2,
	}
	var modelCfg wukong.ModelConfig
	for freq := 32; freq >= 1; freq /= 2 {
		modelCfg.GNN = append(modelCfg.GNN, wukong.GNNModel{
			Freq:      freq,
			ModelPath: "logs/uav/gnn/model",
		})
		modelCfg.Filters = append(modelCfg.Filters, wukong.FilterModel{
			Name: "rnn",
			Freq: freq,
			Cfg: map[string]string{
				"model_path": fmt.Sprintf("logs/%s/%d/filter-rnn/model", predName, freq),
			},
		})
		modelCfg.Refiners = append(modelCfg.Refiners, wukong.RefineModel{
			Name: "rnn",
			Freq: freq,
			Cfg: map[string]string{
				"model_path": fmt.Sprintf("logs/%s/%d/refine-rnn/model", predName, freq),
			},
		})
	}
	return ppCfg, modelCfg
}
