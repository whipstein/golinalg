package golapack

import "testing"

func TestGetIter(t *testing.T) {
	first, last, inc := 1, 10, 1
	exemplar := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	result := genIter(first, last, inc)
	for i, val := range result {
		if val != exemplar[i] {
			t.Errorf("genIter: got %v want %v\n", val, exemplar[i])
		}
	}

	first, last, inc = 10, 1, -1
	exemplar = []int{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
	result = genIter(first, last, inc)
	for i, val := range result {
		if val != exemplar[i] {
			t.Errorf("genIter: got %v want %v\n", val, exemplar[i])
		}
	}
}
