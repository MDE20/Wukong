package gnn

import (
	"../wukong"

	"fmt"
	"sort"
)

type Edge struct {
	LeftFrame  int
	LeftIdx    int
	RightFrame int
	RightIdx   int
	Score      float64
}

func GetEdgeMaps(edges []Edge) (map[[2]int][]Edge, map[[2]int][]Edge) {
	leftMap := make(map[[2]int][]Edge)
	rightMap := make(map[[2]int][]Edge)
	for _, edge := range edges {
		lk := [2]int{edge.LeftFrame, edge.LeftIdx}
		rk := [2]int{edge.RightFrame, edge.RightIdx}
		leftMap[lk] = append(leftMap[lk], edge)
		if edge.RightIdx != -1 {
			rightMap[rk] = append(rightMap[rk], edge)
		}
	}
	return leftMap, rightMap
}

// Update an existing edge set at the specified frames.
// It is expected that the existing edges are captured at a higher (less accurate) freq.
// 根据某些输入帧的推理结果，更新图中的边
func (gnn *GNN) Update(edges []Edge, frames [][2]int, q map[int]float64) []Edge {
	leftMap, rightMap := GetEdgeMaps(edges)
	edgeSet := make(map[Edge]bool)
	for _, edge := range edges {
		edgeSet[edge] = true
	}

	sort.Slice(frames, func(i, j int) bool {
		return frames[i][0] < frames[j][0]
	})

	frameInferList := make([][2]int, len(frames))
	for i, frameSpec := range frames {
		idx1 := frameSpec[0]
		freq := frameSpec[1]
		frameInferList[i] = [2]int{idx1, idx1 + freq}
	}
	mats := gnn.InferMany(frameInferList, "[gnn-update]")

	for i, frameSpec := range frames {
		idx1 := frameSpec[0]
		freq := frameSpec[1]
		idx2 := idx1 + freq
		mat := mats[i]
		// 检查 q[freq] 是否存在，若不存在则设置默认值
		if _, exists := q[freq]; !exists {
			q[freq] = 1.0 // 或者其他默认值
		}

		if q[freq] == 0 {
			panic(fmt.Errorf("gnn update got freq %d without q on frames (%d,%d)", freq, idx1, idx2))
		}
		if len(gnn.detections[idx1]) == 0 || len(gnn.detections[idx2]) == 0 {
			continue
		}
		termIdx := len(gnn.detections[idx2])

		// new edges, grouped by the left detection
		newEdges := make(map[[2]int][]Edge)
		if q[freq] < 1 {
			for i := range mat {
				var maxProb float64 = 0
				for j := range mat[i] {
					if mat[i][j] > maxProb {
						maxProb = mat[i][j] // 找到最大概率
					}
				}
				for j := range mat[i] {
					if mat[i][j] < q[freq]*maxProb {
						continue // 过滤低于阈值的概率
					}
					var edge Edge
					if j == termIdx {
						edge = Edge{idx1, i, idx2, -1, mat[i][j]}
					} else {
						edge = Edge{idx1, i, idx2, j, mat[i][j]}
					}
					lk := [2]int{idx1, i}
					newEdges[lk] = append(newEdges[lk], edge)
				}
			}
		} else {
			// for each right detection, find highest prob left detection
			// then, for each left detection, among those find the best right detection
			fwProbs := make(map[int]float64)
			fwEdges := make(map[int]int)
			for j := 0; j < len(gnn.detections[idx2]); j++ {
				var maxProb float64
				var maxI int
				for i := 0; i < len(gnn.detections[idx1]); i++ {
					if mat[i][j] > maxProb {
						maxProb = mat[i][j]
						maxI = i
					}
				}
				if maxProb > fwProbs[maxI] {
					fwProbs[maxI] = maxProb
					fwEdges[maxI] = j
				}
			}
			for i := 0; i < len(gnn.detections[idx1]); i++ {
				var edge Edge
				if fwProbs[i] > 0 && fwProbs[i] > mat[i][termIdx] {
					edge = Edge{idx1, i, idx2, fwEdges[i], fwProbs[i]}
				} else {
					edge = Edge{idx1, i, idx2, -1, mat[i][termIdx]}
				}
				lk := [2]int{idx1, i}
				newEdges[lk] = append(newEdges[lk], edge)
			}
		}

		// incorporate the edges
		// we do not incorporate ->term edges if there is an existing edge that
		//   matches the left detection (we also anyway do not add these into the
		//   graph unless there is uncertainty)
		// when incorporating, we may need to remove existing edges if there
		//   are conflicts
		for lk, edges := range newEdges {
			isZero := len(edges) == 1 && edges[0].RightIdx == -1
			if isZero && len(leftMap[lk]) == 1 && leftMap[lk][0].RightIdx != -1 {
				continue
			}
			for _, old := range leftMap[lk] {
				delete(edgeSet, old)
			}
			if isZero {
				continue
			}
			for _, edge := range edges {
				rk := [2]int{edge.RightFrame, edge.RightIdx}
				for _, old := range rightMap[rk] {
					delete(edgeSet, old)
				}
				edgeSet[edge] = true
			}
		}
	}

	edges = nil
	for edge := range edgeSet {
		edges = append(edges, edge)
	}
	return edges
}

func (gnn *GNN) GetComponents(edges []Edge) [][]Edge {
	leftMap, rightMap := GetEdgeMaps(edges)

	seen := make(map[Edge]bool)
	getComponent := func(start Edge) []Edge {
		component := []Edge{start}
		q := []Edge{start}
		seen[start] = true
		for len(q) > 0 {
			edge := q[0]
			q = q[1:]
			var others []Edge
			others = append(others, leftMap[[2]int{edge.RightFrame, edge.RightIdx}]...)
			others = append(others, rightMap[[2]int{edge.LeftFrame, edge.LeftIdx}]...)
			for _, other := range others {
				if seen[other] {
					continue
				}
				seen[other] = true
				component = append(component, other)
				q = append(q, other)
			}
		}
		return component
	}

	var components [][]Edge
	for _, edge := range edges {
		if seen[edge] {
			continue
		}
		component := getComponent(edge)
		components = append(components, component)
	}

	return components
}

// Given a concrete component (no ambiguous edges), returns the corresponding
// track. If not concrete, this returns arbitrary track represented by the
// component.
func (gnn *GNN) ComponentToTrack(comp []Edge) []wukong.Detection {
	smallestEdge := comp[0]
	for _, edge := range comp {
		if edge.LeftFrame < smallestEdge.LeftFrame {
			smallestEdge = edge
		}
	}
	leftMap, _ := GetEdgeMaps(comp)
	track := []wukong.Detection{gnn.detections[smallestEdge.LeftFrame][smallestEdge.LeftIdx]}
	last := [2]int{smallestEdge.LeftFrame, smallestEdge.LeftIdx}
	for len(leftMap[last]) > 0 {
		edge := leftMap[last][0]
		if edge.RightIdx == -1 {
			break
		}
		track = append(track, gnn.detections[edge.RightFrame][edge.RightIdx])
		last = [2]int{edge.RightFrame, edge.RightIdx}
	}
	return track
}

func (gnn *GNN) SampleComponent(comp []Edge) [][]wukong.Detection {
	// right now we don't do the sampling
	// actually we just pick the highest scoring one:
	// (1) for each left detection, remove all but the highest outgoing edge
	// (2) then, for each right detection, remove all but the highest incoming edge
	leftMap, _ := GetEdgeMaps(comp)
	edgeSet := make(map[Edge]bool)
	for _, edge := range comp {
		edgeSet[edge] = true
	}
	for _, group := range leftMap {
		if len(group) <= 1 {
			continue
		}
		var bestScore float64 = 0
		for _, edge := range group {
			if edge.Score > bestScore {
				bestScore = edge.Score
			}
		}
		for _, edge := range group {
			if edge.Score < bestScore {
				delete(edgeSet, edge)
			}
		}
	}

	// convert back to list for step 2
	var edges []Edge
	for edge := range edgeSet {
		edges = append(edges, edge)
	}
	_, rightMap := GetEdgeMaps(edges)
	for _, group := range rightMap {
		if len(group) <= 1 {
			continue
		}
		var bestScore float64 = 0
		for _, edge := range group {
			if edge.Score > bestScore {
				bestScore = edge.Score
			}
		}
		for _, edge := range group {
			if edge.Score < bestScore {
				delete(edgeSet, edge)
			}
		}
	}

	// finally, enumerate all the tracks represented in this component
	// there may be more than one, but they will be disjoint
	edges = nil
	for edge := range edgeSet {
		edges = append(edges, edge)
	}
	var tracks [][]wukong.Detection
	for _, subcomp := range gnn.GetComponents(edges) {
		tracks = append(tracks, gnn.ComponentToTrack(subcomp))
	}
	return tracks
}

func (gnn *GNN) GetUncertainFrames(components [][]Edge, seen []int) []int {
	getPrevFrame := func(frameIdx int) int {
		prev := -1
		for _, idx := range seen {
			if idx < frameIdx && (idx > prev || prev == -1) {
				prev = idx
			}
		}
		return prev
	}
	getNextFrame := func(frameIdx int) int {
		next := -1
		for _, idx := range seen {
			if idx > frameIdx && (idx < next || next == -1) {
				next = idx
			}
		}
		return next
	}

	frameSet := make(map[int]bool)
	for _, comp := range components {
		leftMap, rightMap := GetEdgeMaps(comp)
		for lk, group := range leftMap {
			if len(group) <= 1 {
				continue
			}
			next := getNextFrame(lk[0])
			if next == -1 || next-lk[0] <= 1 {
				panic(fmt.Errorf("issue finding frame for uncertain edge"))
			}
			mid := (lk[0] + next) / 2
			frameSet[mid] = true
		}
		for rk, group := range rightMap {
			if len(group) <= 1 {
				continue
			}
			prev := getPrevFrame(rk[0])
			if prev == -1 || rk[0]-prev <= 1 {
				panic(fmt.Errorf("issue finding frame for uncertain edge"))
			}
			mid := (prev + rk[0]) / 2
			frameSet[mid] = true
		}
	}

	var frames []int
	for frameIdx := range frameSet {
		frames = append(frames, frameIdx)
	}
	sort.Ints(frames)
	return frames
}

// UpdateEdge 根据输入的边集和帧数据更新GNN图结构中的边。
// edges: 当前边的集合。
// frames: 包含帧索引和频率的数组，每个元素代表要处理的一对帧及其频率。
// q: 用于过滤边的概率阈值。
func (gnn *GNN) UpdateEdges(edges []Edge, frames [][2]int, q map[int]float64) []Edge {
	// 构建边的索引映射，以便快速查找左侧和右侧节点的边。
	leftMap, rightMap := GetEdgeMaps(edges)

	// 创建一个边集合来存储所有当前的边。
	edgeSet := make(map[Edge]bool)
	for _, edge := range edges {
		edgeSet[edge] = true
	}

	// 对帧进行排序，以确保按照时间顺序进行更新。
	sort.Slice(frames, func(i, j int) bool {
		return frames[i][0] < frames[j][0]
	})

	// 构造推理帧列表，以便后续批量推理。
	frameInferList := make([][2]int, len(frames))
	for i, frameSpec := range frames {
		idx1 := frameSpec[0]
		freq := frameSpec[1]
		frameInferList[i] = [2]int{idx1, idx1 + freq}
	}
	// 使用GNN推理多个帧之间的边。
	mats := gnn.InferMany(frameInferList, "[gnn-update]")

	// 遍历每个帧对，更新边的信息。
	for i, frameSpec := range frames {
		idx1 := frameSpec[0]
		freq := frameSpec[1]
		idx2 := idx1 + freq
		mat := mats[i]

		// 检查并设置缺省概率阈值。
		if _, exists := q[freq]; !exists {
			q[freq] = 1.0
		}

		// 如果阈值为0，直接报错退出。
		if q[freq] == 0 {
			panic(fmt.Errorf("gnn update got freq %d without q on frames (%d,%d)", freq, idx1, idx2))
		}
		// 如果当前帧没有检测结果，则跳过。
		if len(gnn.detections[idx1]) == 0 || len(gnn.detections[idx2]) == 0 {
			continue
		}
		termIdx := len(gnn.detections[idx2])

		// 用于存储新生成的边。
		newEdges := make(map[[2]int][]Edge)

		// 根据阈值 q 进行边的过滤与构建。
		if q[freq] < 1 {
			for i := range mat {
				var maxProb float64 = 0
				for j := range mat[i] {
					if mat[i][j] > maxProb {
						maxProb = mat[i][j] // 找到最大概率值。
					}
				}
				for j := range mat[i] {
					if mat[i][j] < q[freq]*maxProb {
						continue // 过滤低于阈值的边。
					}
					var edge Edge
					if j == termIdx {
						edge = Edge{idx1, i, idx2, -1, mat[i][j]}
					} else {
						edge = Edge{idx1, i, idx2, j, mat[i][j]}
					}
					lk := [2]int{idx1, i}
					newEdges[lk] = append(newEdges[lk], edge)
				}
			}
		} else {
			// 如果阈值为1，则选择每个检测结果的最高概率边。
			fwProbs := make(map[int]float64)
			fwEdges := make(map[int]int)
			for j := 0; j < len(gnn.detections[idx2]); j++ {
				var maxProb float64
				var maxI int
				for i := 0; i < len(gnn.detections[idx1]); i++ {
					if mat[i][j] > maxProb {
						maxProb = mat[i][j]
						maxI = i
					}
				}
				if maxProb > fwProbs[maxI] {
					fwProbs[maxI] = maxProb
					fwEdges[maxI] = j
				}
			}
			for i := 0; i < len(gnn.detections[idx1]); i++ {
				var edge Edge
				if fwProbs[i] > 0 && fwProbs[i] > mat[i][termIdx] {
					edge = Edge{idx1, i, idx2, fwEdges[i], fwProbs[i]}
				} else {
					edge = Edge{idx1, i, idx2, -1, mat[i][termIdx]}
				}
				lk := [2]int{idx1, i}
				newEdges[lk] = append(newEdges[lk], edge)
			}
		}

		// 将新边合并到边集合中。
		for lk, edges := range newEdges {
			isZero := len(edges) == 1 && edges[0].RightIdx == -1
			if isZero && len(leftMap[lk]) == 1 && leftMap[lk][0].RightIdx != -1 {
				continue
			}
			for _, old := range leftMap[lk] {
				delete(edgeSet, old)
			}
			if isZero {
				continue
			}
			for _, edge := range edges {
				rk := [2]int{edge.RightFrame, edge.RightIdx}
				for _, old := range rightMap[rk] {
					delete(edgeSet, old)
				}
				edgeSet[edge] = true
			}
		}
	}

	// 将更新后的边集合转换为数组返回。
	edges = nil
	for edge := range edgeSet {
		edges = append(edges, edge)
	}
	return edges
}
