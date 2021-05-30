package golapack

import "testing"

func TestMaxlocf64(t *testing.T) {
	test := []float64{3, -5, -4.5, 3.2, 9.4, 8.6}
	maxloc := 5
	result := maxlocf64(test...)
	if result != maxloc {
		t.Errorf("maxlocf64: got %v want %v\n", result, maxloc)
	}
}
