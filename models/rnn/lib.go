package rnn

import (
	"github.com/zhengpeijun/wukong-master/wukong"
	"github.com/zhengpeijun/wukong-master/predicate"

	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os/exec"
	"sort"
	"strconv"
)

type Item struct {
	Track []wukong.Detection `json:"track"`
	Label []int             `json:"label"`
}

type DS struct {
	Train []Item `json:"train"`
	Val   []Item `json:"val"`
}

func boolToInt(b []bool) []int {
	x := make([]int, len(b))
	for i := range x {
		if b[i] {
			x[i] = 1
		}
	}
	return x
}

type Model struct {
	cmd   *exec.Cmd
	stdin io.WriteCloser
	rd    *bufio.Reader
}

func MakeModel(numOutputs int, modelPath string) Model {
	fmt.Println("[ThismodelPath] cfg:", modelPath)
	cmd, stdin, stdout := wukong.Command("rnn-model", "python", "models/rnn/wrapper.py", strconv.Itoa(numOutputs), modelPath)
	rd := bufio.NewReader(stdout)
	return Model{cmd, stdin, rd}
}

func (m Model) Infer(tracks [][]wukong.Detection) [][]float64 {
	bytes, err := json.Marshal(tracks)
	if err != nil {
		panic(err)
	}
	if _, err := m.stdin.Write([]byte(string(bytes) + "\n")); err != nil {
		panic(err)
	}
	line, err := m.rd.ReadString('\n')
	if err != nil {
		panic(err)
	}
	var outputs [][]float64
	if err := json.Unmarshal([]byte(line), &outputs); err != nil {
		panic(err)
	}
	return outputs
}

func (m Model) Close() {
	m.stdin.Close()
	m.cmd.Wait()
}

// Capture the track at the specified prefix freq and suffix freq.
// There are two use cases:
// (a) Create an input for rnn-ps-refine. In this case, addFake=true, and if
//
//	there are missing detections at the prefix/suffix, we copy the first or
//	last detection in the coarse track to fill it in. This mimics what the
//	rnn-ps-refine method does at inference time.
//
// (b) Test whether further prefix or suffix refinement will help. In this case,
//
//	addFake=false, and either pFreq or sFreq should be 1. We will not add any
//	fake filler detections. But we will capture all intermediate detections
//	rather than getting it coarsely, and only either prefix or suffix will be
//	missing some detections.
func GetCoarsePSRefine(track []wukong.Detection, freq int, pFreq int, sFreq int, k int, addFake bool) []wukong.Detection {
	var coarse []wukong.Detection
	start := -1
	end := -1
	frameToDetection := make(map[int]wukong.Detection)
	for i, detection := range track {
		frameToDetection[detection.FrameIdx] = detection
		if detection.FrameIdx%freq != k {
			continue
		}
		coarse = append(coarse, detection)
		if start == -1 {
			start = i
		}
		end = i
	}
	if len(coarse) == 0 {
		return coarse
	}
	if !addFake {
		// this suggests case (b) above, so we add in all the intermediate detections
		coarse = append([]wukong.Detection{}, track[start:end+1]...)
	}
	var fakes []wukong.Detection
	for x := freq / 2; x >= pFreq; x /= 2 {
		frameIdx := coarse[0].FrameIdx - x
		detection, ok := frameToDetection[frameIdx]
		if ok {
			coarse = append([]wukong.Detection{detection}, coarse...)
		} else {
			fake := wukong.Detection{
				FrameIdx: frameIdx,
				Left:     0,
				Top:      0,
				Right:    0,
				Bottom:   0,
			}
			fakes = append(fakes, fake)
		}
	}
	for x := freq / 2; x >= sFreq; x /= 2 {
		frameIdx := coarse[len(coarse)-1].FrameIdx + x
		detection, ok := frameToDetection[frameIdx]
		if ok {
			coarse = append(coarse, detection)
		} else {
			fake := wukong.Detection{
				FrameIdx: frameIdx,
				Left:     0,
				Top:      0,
				Right:    0,
				Bottom:   0,
			}
			fakes = append(fakes, fake)
		}
	}
	if addFake {
		coarse = append(coarse, fakes...)
		sort.Slice(coarse, func(i, j int) bool {
			return coarse[i].FrameIdx < coarse[j].FrameIdx
		})
	}
	return coarse
}

func ItemsFromSegments(segments []wukong.Segment, freq int, predFunc predicate.Predicate) ([]Item, []Item) {
	var tracks [][]wukong.Detection
	log.Printf("[freq]  (freq=%d )", freq)
	for _, segment := range segments {
		segTracks := wukong.GetTracks(wukong.ReadDetections(segment.TrackPath))
		tracks = append(tracks, segTracks...)
	}

	labels := make([]bool, len(tracks))
	var numTrue, numFalse int
	for i, track := range tracks {
		if predFunc([][]wukong.Detection{track}) {
			labels[i] = true
			numTrue++
		} else {
			labels[i] = false
			numFalse++
		}
	}

	// for rnn-ps-refine, we will sample random freq, since we want the rnn to tell
	//   us whether to continue refining after the first few refinements and etc.
	var freqList []int
	for x := freq; x >= 2; x /= 2 {
		freqList = append(freqList, x)
	}

	coarsePerTrueTrack := 1 + (50000 / numTrue)
	coarsePerFalseTrack := 1 + (50000 / numFalse)
	log.Printf("[prepare] compute coarse tracks (max true=%d false=%d per track with %d tracks)", coarsePerTrueTrack, coarsePerFalseTrack, len(tracks))
	var filterItems, refineItems []Item
	for i, track := range tracks {
		var n int
		if labels[i] {
			n = coarsePerTrueTrack
		} else {
			n = coarsePerFalseTrack
		}
		for k := 0; k < freq && k < n; k++ {
			coarse := wukong.GetCoarse(track, freq, k)

			filterItems = append(filterItems, Item{
				Track: coarse,
				Label: boolToInt([]bool{labels[i]}),
			})
		}

		// items for refinement
		if !labels[i] {
			continue
		}
		for iter := 0; iter < n; iter++ {
			pFreq := freqList[rand.Intn(len(freqList))]
			sFreq := freqList[rand.Intn(len(freqList))]
			k := rand.Intn(freq)
			coarse := GetCoarsePSRefine(track, freq, pFreq, sFreq, k, true)
			noPrefix := GetCoarsePSRefine(track, freq, pFreq, 1, k, false)
			noSuffix := GetCoarsePSRefine(track, freq, 1, sFreq, k, false)
			needsPrefix := !predFunc([][]wukong.Detection{noPrefix})
			needsSuffix := !predFunc([][]wukong.Detection{noSuffix})
			refineItems = append(refineItems, Item{
				Track: coarse,
				Label: boolToInt([]bool{needsPrefix, needsSuffix}),
			})
		}
	}

	return filterItems, refineItems
}
