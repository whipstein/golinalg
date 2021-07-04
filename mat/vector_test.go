package mat

import (
	"testing"
)

// func TestVectorBuilder(t *testing.T) {
// 	vb := NewVectorBuilder()
// 	x := vb.Data(make([]float64, 10)).Build()
// 	for i := 0; i < x.Size(); i++ {
// 		x.Set(i, float64(i))
// 	}

// 	if x.Size() != 10 {
// 		t.Errorf("VectorBuilder failed\n")
// 	}

// 	y := []float64{0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4}
// 	x = vb.Data(y).Build()

// 	for i, val := range y {
// 		if x.Get(i) != val {
// 			t.Errorf("VectorBuilder failed: got %v  want %v\n", x.Get(i), val)
// 		}
// 	}
// }

func TestVector(t *testing.T) {
	vf := VectorFactory()
	vdf := VectorDataFactory()
	x := vf(10)
	xx := vf(20, 2)
	xxx := vf(30, -3)
	for i := 0; i < x.Size; i++ {
		x.Set(i, float64(i+1))
		xx.Set(i, float64(i+1))
		xxx.Set(i, float64(i+1))
	}
	if x.Size != 10 || xx.Size != 20 || xxx.Size != 30 || len(x.Data) != 10 || len(xx.Data) != 20 || len(xxx.Data) != 30 {
		t.Errorf("VectorFactory failed\n")
	}

	y := []float64{0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4}
	yy := append(y, 1.6, 1.1, -1.5, 1.8, 1.9, -1.3, -1.4)
	x = vdf(y)
	xx = vdf(yy, 2)
	xxx = vdf(yy, -2)
	ixx := xx.Iter(len(y))
	ixxx := xxx.Iter(len(y))

	// Setup data that uses same memory address for multiple vectors
	for i, val := range y {
		if x.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", x.Get(i), val)
		}
		if xx.Get(ixx[i]) != yy[i*xx.Inc] || xx.Data[i] != yy[i] {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xx.Get(ixx[i]), yy[i*xx.Inc])
		}
		if xxx.Get(ixxx[i]) != yy[(len(y)-1-i)*absint(xxx.Inc)] || xxx.Data[i] != yy[i] {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xxx.Get(ixxx[i]), yy[(len(y)-1-i)*absint(xxx.Inc)])
		}
		x.Set(i, 0.5*x.Get(i))
		xx.Set(i, 0.5*xx.Get(i))
	}
	for i, val := range y {
		if x.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", x.Get(i), val)
		}
		if xx.Get(ixx[i]) != yy[i*xx.Inc] || xx.Data[i] != yy[i] {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xx.Get(ixx[i]), yy[i*xx.Inc])
		}
		if xxx.Get(ixxx[i]) != yy[(len(y)-1-i)*absint(xxx.Inc)] || xxx.Data[i] != yy[i] {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xxx.Get(ixxx[i]), yy[(len(y)-1-i)*absint(xxx.Inc)])
		}
	}

	// Setup data that uses separate memory addresses for multiple vectors
	x = vf(len(y))
	xx = vf(len(yy), 2)
	xxx = vf(len(yy), -2)
	ixx = xx.Iter(len(y))
	ixxx = xxx.Iter(len(y))
	for i, val := range y {
		x.Set(i, val)
		xx.Set(i, val)
		xxx.Set(i, val)
	}
	for i, val := range y {
		if x.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", x.Get(i), val)
		}
		x.Set(i, 10*x.Get(i))
		if xx.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xx.Get(i), val)
		}
		xx.Set(i, 10*xx.Get(i))
		if xxx.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xxx.Get(i), val)
		}
		xxx.Set(i, 10*xxx.Get(i))
	}
	for i, val := range y {
		if x.Get(i) == val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", x.Get(i), val)
		}
		if xx.Get(ixx[i]) == yy[i*xx.Inc] {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xx.Get(ixx[i]), yy[i*xx.Inc])
		}
		if xxx.Get(ixxx[i]) == yy[(len(y)-1-i)*absint(xxx.Inc)] {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xxx.Get(ixxx[i]), yy[(len(y)-1-i)*absint(xxx.Inc)])
		}
	}
}

func BenchmarkVectorSet(b *testing.B) {
	vf := VectorFactory()
	dx := vf(b.N)
	da := 0.4

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dx.Set(i, da)
	}
}

func BenchmarkCVectorSet(b *testing.B) {
	cvf := CVectorFactory()
	cx := cvf(b.N)
	ca := 0.4 - 0.7i

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cx.Set(i, ca)
	}
}
