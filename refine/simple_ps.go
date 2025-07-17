package refine

import (
	"../wukong"
	"../predicate"

	"fmt"
	"sort"
	"strconv"
)

// Get coarse with all intermediate detections, just missing prefix and suffix.
func GetCoarsePS(track []wukong.Detection, freq int, k int) []wukong.Detection {
	start := -1
	end := -1
	for i, detection := range track {
		if detection.FrameIdx%freq != k {
			continue
		}
		if start == -1 {
			start = i
		}
		end = i
	}

	_ = fmt.Sprintf("CoarsePS from %d to %d (k=%d)", start, end, k)
	if start == -1 || end == -1 {
		return nil
	}
	return track[start : end+1]
}

type SimplePSRefiner struct {
	freq          int
	predFunc      predicate.Predicate
	freqThreshold int

	debugMode     bool
	refineNotes   string
	lastUsedBound float64
}

func MakeSimplePSRefiner(freq int, trainTracks [][]wukong.Detection, predFunc predicate.Predicate, modelCfg map[string]string, cfg map[string]string) Refiner {
	r := &SimplePSRefiner{
		freq:     freq,
		predFunc: predFunc,
		debugMode:   false, 
		refineNotes: "Initialized",
	}
	if cfg["threshold"] != "" {
		var err error
		r.freqThreshold, err = strconv.Atoi(cfg["threshold"])
		if err != nil {
			panic(err)
		}
	}

	if r.debugMode {
		fmt.Println("[DEBUG] Created SimplePSRefiner with freq =", r.freq)
	}
	return r
}

func init() {
	PSRefiners["simple"] = MakeSimplePSRefiner
}

func (r *SimplePSRefiner) Plan(valTracks [][]wukong.Detection, bound float64) map[string]string {
	// for each coarse track, find the freqThreshold needed to get the predicate correct
	// (but retain all intermediate detections in the coarse tracks)
	// then choose a threshold based on bounds
	var samples []int
	r.lastUsedBound = bound
	for _, track := range valTracks {
		label := r.predFunc([][]wukong.Detection{track})
		if !label {
			// negative->positive due to coarse is unlikely
			continue
		}
		for k := 0; k < r.freq; k++ {
			freqThreshold := r.freq
			for {
				coarse := GetCoarsePS(track, freqThreshold, k%freqThreshold)
				if r.debugMode {
					fmt.Printf("[DEBUG] Checking coarse track (k=%d, freq=%d)\n", k, freqThreshold)
				}
				if r.predFunc([][]wukong.Detection{coarse}) == label {
					break
				}
				freqThreshold /= 2
				if freqThreshold < 1 {
					panic(fmt.Errorf("simple ps planner: freqThreshold==1 should always succeed"))
				}
			}
			samples = append(samples, freqThreshold)
		}
	}
	sort.Ints(samples)

	if len(samples) > 0 && samples[0] > samples[len(samples)-1] {
		_ = fmt.Sprintf("[WARN] Samples are not sorted!") 
	}

	r.freqThreshold = samples[int((1-bound)*float64(len(samples)))]

	if r.debugMode {
		fmt.Printf("[DEBUG] Planned threshold = %d (bound = %.2f)\n", r.freqThreshold, bound)
	}
	return map[string]string{
		"threshold": fmt.Sprintf("%d", r.freqThreshold),
	}
}

func (r *SimplePSRefiner) Step(tracks [][]wukong.Detection, seen []int) ([]int, []int) {
	seenSet := make(map[int]bool)
	for _, frameIdx := range seen {
		seenSet[frameIdx] = true
	}

	getFreq := func(frameIdx int) int {
		for freq := r.freq; freq >= 2; freq /= 2 {
			if frameIdx%freq == 0 {
				return freq
			}
		}
		return 1
	}

	// Get the next frame idx that we need to look at.
	find := func(frameIdx int, direction int) int {
		freq := getFreq(frameIdx)
		if r.debugMode {
			fmt.Printf("[DEBUG] find() called on frame %d dir %d, initial freq %d\n", frameIdx, direction, freq)
		}
		for seenSet[frameIdx] {
			freq = freq / 2
			if freq < r.freqThreshold {
				return -1
			}
			frameIdx = frameIdx + direction*freq
		}
		return frameIdx
	}

	needed := make(map[int]bool)
	var refined []int
	for i, track := range tracks {
		if r.predFunc([][]wukong.Detection{track}) {
			continue
		}
		prefixIdx := find(track[0].FrameIdx, -1)
		suffixIdx := find(track[len(track)-1].FrameIdx, 1)
		if prefixIdx != -1 {
			needed[prefixIdx] = true
		}
		if suffixIdx != -1 {
			needed[suffixIdx] = true
		}
		if prefixIdx != -1 || suffixIdx != -1 {
			refined = append(refined, i)
		}
	}
	if r.debugMode {
		fmt.Printf("[DEBUG] Refined %d tracks, Needed %d new frames\n", len(refined), len(needed))
	}
	var neededList []int
	for frameIdx := range needed {
		neededList = append(neededList, frameIdx)
	}
	return neededList, refined
}

func (r *SimplePSRefiner) Close() {
	if r.debugMode {
		fmt.Println("[DEBUG] SimplePSRefiner closing.")
	}
}

func (r *SimplePSRefiner) AdjustParameters(paramAdjustmentFactor float64) {
	fmt.Printf("Old freqThreshold: %d\n", r.freqThreshold)
	newThreshold := int(float64(r.freqThreshold) * paramAdjustmentFactor)

	if newThreshold < 1 {
		newThreshold = 1
		fmt.Println("[WARN] freqThreshold clamped to minimum 1")
	}
	if newThreshold > 16 {
		newThreshold = 16
		fmt.Println("[WARN] freqThreshold clamped to maximum 64")
	}

	r.freqThreshold = newThreshold

	fmt.Printf("Adjusted freqThreshold to: %d\n", r.freqThreshold)
}
