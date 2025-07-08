package exec

import (
	filterlib "github.com/zhengpeijun/miris-master/filter"
	gnnlib "github.com/zhengpeijun/miris-master/gnn"
	"github.com/zhengpeijun/miris-master/miris"
	"github.com/zhengpeijun/miris-master/predicate"
	"github.com/zhengpeijun/miris-master/refine"

	"log"
	"os"
	"sort"
)

// Returns needed [2]int{frameIdx, freq} from a list of frames that we need
func getNeededSpecs(needed []int, seenFrames map[int]bool, maxFrame int) [][2]int {
	var frames [][2]int
	for _, frameIdx := range needed {
		if seenFrames[frameIdx] || frameIdx < 0 || frameIdx > maxFrame {
			continue
		}
		idx1 := -1
		idx2 := -1
		for seenIdx := range seenFrames {
			if seenIdx < frameIdx && (idx1 == -1 || seenIdx > idx1) {
				idx1 = seenIdx
			} else if seenIdx > frameIdx && (idx2 == -1 || seenIdx < idx2) {
				idx2 = seenIdx
			}
		}
		freq1 := frameIdx - idx1
		freq2 := idx2 - frameIdx
		frames = append(frames, [2]int{idx1, freq1})
		frames = append(frames, [2]int{frameIdx, freq2})
	}
	for _, frameSpec := range frames {
		seenFrames[frameSpec[0]] = true
		seenFrames[frameSpec[0]+frameSpec[1]] = true
	}
	return frames
}

type GraphWithSeen struct {
	Graph []gnnlib.Edge
	Seen  map[int]bool
}

func ReadGraphAndSeen(fname string) ([]gnnlib.Edge, map[int]bool) {
	var x GraphWithSeen
	miris.ReadJSON(fname, &x)
	return x.Graph, x.Seen
}

func Exec(ppCfg miris.PreprocessConfig, modelCfg miris.ModelConfig, plan miris.PlannerConfig, execCfg miris.ExecConfig) {
	predFunc := predicate.GetPredicate(ppCfg.Predicate)
	log.Printf("[exec] loading train tracks")
	var trainTracks [][]miris.Detection
	for _, segment := range ppCfg.TrainSegments {
		segDetections := miris.ReadDetections(segment.TrackPath)
		segTracks := miris.GetTracks(segDetections)
		trainTracks = append(trainTracks, segTracks...)
	}
	trainLabels := make([]bool, len(trainTracks))
	for i, track := range trainTracks {
		trainLabels[i] = predFunc([][]miris.Detection{track})
	}

	log.Printf("[exec] initializing models")
	filter := filterlib.FilterMap[plan.Filter.Name](plan.Freq, trainTracks, trainLabels, modelCfg.GetFilterCfg(plan.Filter.Name, plan.Freq))
	defer filter.Close()

	var gnnPath string
	for _, gnnCfg := range modelCfg.GNN {
		if gnnCfg.Freq != plan.Freq {
			continue
		}
		gnnPath = gnnCfg.ModelPath
	}
	gnn := gnnlib.NewGNN(gnnPath, execCfg.DetectionPath, execCfg.FramePath, ppCfg.FrameScale)
	defer gnn.Close()

	r1 := refine.PSRefiners[plan.Refine.PSMethod](plan.Freq, trainTracks, predFunc, modelCfg.GetRefineCfg(plan.Refine.PSMethod, plan.Freq), plan.Refine.PSCfg)
	defer r1.Close()
	r2 := refine.InterpRefiners[plan.Refine.InterpMethod](plan.Freq, trainTracks, predFunc, modelCfg.GetRefineCfg(plan.Refine.InterpMethod, plan.Freq), plan.Refine.InterpCfg)
	defer r2.Close()
	refiners := []refine.Refiner{r1, r2}

	seenFrames := make(map[int]bool)
	var graph []gnnlib.Edge
	if _, err := os.Stat(execCfg.TrackOutput); err != nil {
		log.Printf("[exec] run initial tracking")

		for _, freq := range []int{2 * plan.Freq, plan.Freq} {
			var frames [][2]int
			for frameIdx := 0; frameIdx < gnn.NumFrames()-freq; frameIdx += freq {
				frames = append(frames, [2]int{frameIdx, freq})
				seenFrames[frameIdx] = true
				seenFrames[frameIdx+freq] = true
			}
			graph = gnn.Update(graph, frames, plan.Q)
		}
		miris.WriteJSON(execCfg.TrackOutput, GraphWithSeen{graph, seenFrames})
	} else {
		log.Printf("[exec] read track output")
		graph, seenFrames = ReadGraphAndSeen(execCfg.TrackOutput)
	}
	log.Printf("[exec] ... tracking yields graph with %d edges (seen %d frames)", len(graph), len(seenFrames))
	maxFrame := ((gnn.NumFrames() - 1) / plan.Freq) * plan.Freq

	// now filter the components
	var components [][]gnnlib.Edge
	if _, err := os.Stat(execCfg.FilterOutput); err != nil {
		log.Printf("[exec] run filtering")
		allComponents := gnn.GetComponents(graph)

		var tracks [][]miris.Detection
		var trackToComponent []int
		for i, comp := range allComponents {
			for _, track := range gnn.SampleComponent(comp) {
				tracks = append(tracks, track)
				trackToComponent = append(trackToComponent, i)
			}
		}

		scores := filter.Predict(tracks)
		goodComponents := make(map[int]bool)
		for j, score := range scores {
			if score < plan.Filter.Threshold {
				continue
			}
			goodComponents[trackToComponent[j]] = true
		}

		for compIdx := range goodComponents {
			components = append(components, allComponents[compIdx])
		}

		miris.WriteJSON(execCfg.FilterOutput, components)
	} else {
		log.Printf("[exec] read filter output")
		miris.ReadJSON(execCfg.FilterOutput, &components)
	}
	log.Printf("[exec] ... got %d good components after filtering", len(components))

	// create helper to select only components that intersect with ones that passed filtering
	vertices := make(map[[2]int]bool)
	for _, comp := range components {
		for _, edge := range comp {
			vertices[[2]int{edge.LeftFrame, edge.LeftIdx}] = true
			if edge.RightIdx != -1 {
				vertices[[2]int{edge.RightFrame, edge.RightIdx}] = true
			}
		}
	}
	filterComponents := func(components [][]gnnlib.Edge) [][]gnnlib.Edge {
		var selected [][]gnnlib.Edge
		for _, comp := range components {
			any := false
			for _, edge := range comp {
				if vertices[[2]int{edge.LeftFrame, edge.LeftIdx}] || vertices[[2]int{edge.RightFrame, edge.RightIdx}] {
					any = true
					break
				}
			}
			if any {
				selected = append(selected, comp)
			}
		}
		return selected
	}

	// 定义缓存
	cache := make(map[int][]gnnlib.Edge) // 缓存帧编号与推理结果的映射
	// someMap := make(map[int]float64)

	// uncertainty resolution
	if _, err := os.Stat(execCfg.UncertaintyOutput); err != nil {
		log.Printf("[exec] run uncertainty resolution")
		for iter := 0; ; iter++ {
			// 确定需要的帧
			var seenList []int
			for frameIdx := range seenFrames {
				seenList = append(seenList, frameIdx)
			}
			sort.Ints(seenList)
			components := gnn.GetComponents(graph)
			components = filterComponents(components)
			neededFrames := gnn.GetUncertainFrames(components, seenList)
			log.Printf("[exec-uncertainty] ... require %d frames on iter %d", len(neededFrames), iter)

			// 动态调整频率：设定阈值
			const frameThreshold = 50 // 设定阈值，例如需要处理的帧数量大于10
			if len(neededFrames) < frameThreshold {
				log.Printf("[exec-uncertainty] Skipping detailed update due to low frame requirement")
				break // 如果需要处理的帧数小于阈值，跳出循环
			}

			// 获取所需帧的规格
			frameSpecs := getNeededSpecs(neededFrames, seenFrames, maxFrame)

			// 使用缓存优化推理
			var framesToUpdate [][2]int
			var framesToInfer [][2]int
			processedFrames := make(map[int]bool)

			// 批量处理缓存和推理
			for _, frameSpec := range frameSpecs {
				if processedFrames[frameSpec[0]] {
					continue // 跳过已处理帧
				}

				if _, found := cache[frameSpec[0]]; found {
					framesToUpdate = append(framesToUpdate, frameSpec) // 已缓存的帧，准备更新图
				} else {
					framesToInfer = append(framesToInfer, frameSpec) // 未缓存的帧，准备推理
				}
				processedFrames[frameSpec[0]] = true // 标记已处理
			}

			// 批量推理未命中的帧
			if len(framesToInfer) > 0 {
				// 推理前检查帧数量
				log.Printf("准备推理的帧数量: %d", len(framesToInfer))

				// 调用模型推理
				// results := gnn.Update(nil, framesToInfer, someMap)

				// 更新缓存并记录已处理帧
				var tempFramesToUpdate [][2]int
				for _, frameSpec := range framesToInfer {
					tempFramesToUpdate = append(tempFramesToUpdate, frameSpec)
				}

				framesToUpdate = append(framesToUpdate, tempFramesToUpdate...)

				// 清空推理队列
				framesToInfer = framesToInfer[:0]
			}

			// 统一更新图
			if len(framesToUpdate) > 0 {
				graph = gnn.Update(graph, framesToUpdate, plan.Q)
			}

			// 判断是否还需要继续迭代
			if len(frameSpecs) == 0 || len(neededFrames) == 0 {
				break // 如果没有更多需要处理的帧，退出循环
			}
		}

		miris.WriteJSON(execCfg.UncertaintyOutput, GraphWithSeen{graph, seenFrames})
	} else {
		log.Printf("[exec] read uncertainty resolution output")
		graph, seenFrames = ReadGraphAndSeen(execCfg.UncertaintyOutput)
	}

	log.Printf("[exec] ... got graph with %d edges after uncertainty resolution (seen %d frames)", len(graph), len(seenFrames))

	// 定义 threshold
	const threshold = 26000                 // 可根据需要调整
	var paramAdjustmentFactor float64 = 1.0 // 初始调整因子
	// refinement
	if _, err := os.Stat(execCfg.RefineOutput); err != nil {
		log.Printf("[exec] run refinement")
		for _, r := range refiners {

			var lastEdgeCount int
			for iter := 0; ; iter++ {
				// determine needed frames
				components := gnn.GetComponents(graph)
				components = filterComponents(components)
				inTracks := make([][]miris.Detection, len(components))
				for i, comp := range components {
					inTracks[i] = gnn.ComponentToTrack(comp)
				}
				var seenList []int
				for frameIdx := range seenFrames {
					seenList = append(seenList, frameIdx)
				}
				sort.Ints(seenList)

				// 调整精炼器的参数
				r.AdjustParameters(paramAdjustmentFactor)

				neededFrames, _ := r.Step(inTracks, seenList)
				log.Printf("[exec-refine] ... require %d frames on iter %d", len(neededFrames), iter)
				frameSpecs := getNeededSpecs(neededFrames, seenFrames, maxFrame)
				if len(frameSpecs) == 0 {
					break
				}

				graph = gnn.Update(graph, frameSpecs, plan.Q)

				//动态调整参数依据: 图的边数量变化
				currentEdgeCount := len(graph)
				edgeChange := currentEdgeCount - lastEdgeCount
				log.Printf("edgeChange = %d", edgeChange)
				if edgeChange < threshold { // 如果边数量变化小于某个阈值，则调整参数
					paramAdjustmentFactor *= 1.1 // 增加参数调整因子
					log.Printf("edgeChange < threshold paramAdjustmentFactor = %f", paramAdjustmentFactor)
				} else {
					paramAdjustmentFactor *= 0.9 // 减少参数调整因子
					log.Printf("edgeChange > threshold paramAdjustmentFactor = %f", paramAdjustmentFactor)
				}
				lastEdgeCount = currentEdgeCount
			}
		}

		miris.WriteJSON(execCfg.RefineOutput, GraphWithSeen{graph, seenFrames})
	} else {
		log.Printf("[exec] read refinement output")
		graph, seenFrames = ReadGraphAndSeen(execCfg.RefineOutput)
	}
	log.Printf("[exec] ... got graph with %d edges after refinement (seen %d frames)", len(graph), len(seenFrames))

	// extract tracks
	components = gnn.GetComponents(graph)
	//components = filterComponents(components)
	var tracks [][]miris.Detection
	for _, comp := range components {
		for _, track := range gnn.SampleComponent(comp) {
			for i := range track {
				track[i].TrackID = len(tracks) + 1
			}
			tracks = append(tracks, track)
		}
	}
	log.Printf("[exec] extracted %d tracks", len(tracks))
	miris.WriteJSON(execCfg.OutPath, miris.TracksToDetections(tracks))

	// evaluate predicate
	var num int
	for _, track := range tracks {
		if predFunc([][]miris.Detection{track}) {
			num++
		}
	}
	log.Printf("[exec] %d/%d tracks satisfy the predicate", num, len(tracks))
}
