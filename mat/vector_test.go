package mat

import "testing"

var vf = VectorFactory()
var vdf = VectorDataFactory()
var cvf = CVectorFactory()
var cvdf = CVectorDataFactory()

func TestVector(t *testing.T) {
	x := vf(10)
	xx := vf(20)
	xxx := vf(30)
	for i := 0; i < x.size; i++ {
		x.Set(i, float64(i+1))
		xx.Set(i, float64(i+1))
		xxx.Set(i, float64(i+1))
	}
	if x.size != 10 || xx.size != 20 || xxx.size != 30 || len(x.Data()) != 10 || len(xx.Data()) != 20 || len(xxx.Data()) != 30 {
		t.Errorf("VectorFactory failed\n")
	}

	y := []float64{0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4}
	yy := append(y, 1.6, 1.1, -1.5, 1.8, 1.9, -1.3, -1.4)
	x = vdf(y)
	xx = vdf(yy)
	xxx = vdf(yy)
	ixx := xx.Iter(len(y), 2)
	ixxx := xxx.Iter(len(y), -2)

	// Setup data that uses same memory address for multiple vectors
	for i, val := range y {
		if x.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", x.Get(i), val)
		}
		if xx.Data()[ixx[i]] != yy[i*2] {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xx.Data()[ixx[i]], yy[i*2])
		}
		if xxx.Data()[ixxx[i]] != yy[13-i*2] {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xxx.Data()[ixxx[i]], yy[13-i*2])
		}
		x.Set(i, 0.5*x.Get(i))
		xx.Set(i, 0.5*xx.Get(i))
	}

	// Setup data that uses separate memory addresses for multiple vectors
	x = vf(len(y))
	xx = vf(len(yy))
	xxx = vf(len(yy))
	ixx = xx.Iter(len(y), 2)
	ixxx = xxx.Iter(len(y), -2)
	for i, val := range y {
		x.Set(i, val)
	}
	for i, val := range yy {
		xx.Set(i, val)
		xxx.Set(i, val)
	}
	for i, val := range y {
		if x.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", x.Get(i), val)
		}
		if xx.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xx.Get(i), val)
		}
		if xxx.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xxx.Get(i), val)
		}
	}
	for i, val := range y {
		if x.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", x.Get(i), val)
		}
		if xx.Data()[ixx[i]] != yy[i*2] {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xx.Data()[ixx[i]], yy[i*2])
		}
		if xxx.Data()[ixxx[i]] != yy[13-i*2] {
			t.Errorf("VectorFactory failed: got %v  want %v\n", xxx.Data()[ixxx[i]], yy[13-i*2])
		}
	}
}

func BenchmarkVectorGet(b *testing.B) {
	dx := vf(b.N)
	da := 0.4

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		da = dx.Get(i)
		_ = da
	}
}
func BenchmarkVectorSet(b *testing.B) {
	dx := vf(b.N)
	da := 0.4

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dx.Set(i, da)
	}
}

// func TestCVector(t *testing.T) {
// 	x := cvf(10)
// 	xx := cvf(20, 2)
// 	xxx := cvf(30, -3)
// 	for i := 0; i < x.Size; i++ {
// 		x.Set(i, complex(float64(i+1), -float64(i-1)))
// 		xx.Set(i, complex(float64(i+1), -float64(i-1)))
// 		xxx.Set(i, complex(float64(i+1), -float64(i-1)))
// 	}
// 	if x.Size != 10 || xx.Size != 20 || xxx.Size != 30 || len(x.Data) != 10 || len(xx.Data) != 40 || len(xxx.Data) != 90 {
// 		t.Errorf("CVectorFactory failed\n")
// 	}

// 	y := []complex128{0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4}
// 	yy := append(y, 1.6, 1.1, -1.5, 1.8, 1.9, -1.3, -1.4)
// 	x = cvdf(y)
// 	xx = cvdf(yy, 2)
// 	xxx = cvdf(yy, -2)
// 	ixx := xx.Iter(len(y))
// 	ixxx := xxx.Iter(len(y))

// 	// Setup data that uses same memory address for multiple vectors
// 	for i, val := range y {
// 		if x.Get(i) != val {
// 			t.Errorf("CVectorFactory failed: got %v  want %v\n", x.Get(i), val)
// 		}
// 		if xx.Data[ixx[i]] != yy[i] {
// 			t.Errorf("CVectorFactory failed: got %v  want %v\n", xx.Data[ixx[i]], yy[i])
// 		}
// 		if xxx.Data[ixxx[i]] != yy[i] {
// 			t.Errorf("CVectorFactory failed: got %v  want %v\n", xxx.Data[ixxx[i]], yy[i])
// 		}
// 		x.Set(i, 0.5*x.Get(i))
// 		xx.Set(i, 0.5*xx.Get(i))
// 	}

// 	// Setup data that uses separate memory addresses for multiple vectors
// 	x = cvf(len(y))
// 	xx = cvf(len(yy), 2)
// 	xxx = cvf(len(yy), -2)
// 	ixx = xx.Iter(len(y))
// 	ixxx = xxx.Iter(len(y))
// 	for i, val := range y {
// 		x.Set(i, val)
// 		xx.Set(i, val)
// 		xxx.Set(i, val)
// 	}
// 	for i, val := range y {
// 		if x.Get(i) != val {
// 			t.Errorf("CVectorFactory failed: got %v  want %v\n", x.Get(i), val)
// 		}
// 		if xx.Get(i) != val {
// 			t.Errorf("CVectorFactory failed: got %v  want %v\n", xx.Get(i), val)
// 		}
// 		if xxx.Get(i) != val {
// 			t.Errorf("CVectorFactory failed: got %v  want %v\n", xxx.Get(i), val)
// 		}
// 	}
// 	for i, val := range y {
// 		if x.Get(i) != val {
// 			t.Errorf("CVectorFactory failed: got %v  want %v\n", x.Get(i), val)
// 		}
// 		if xx.Data[ixx[i]] != yy[i] {
// 			t.Errorf("CVectorFactory failed: got %v  want %v\n", xx.Data[ixx[i]], yy[i])
// 		}
// 		if xxx.Data[ixxx[i]] != yy[i] {
// 			t.Errorf("CVectorFactory failed: got %v  want %v\n", xxx.Data[ixxx[i]], yy[i])
// 		}
// 	}
// }

// func BenchmarkCVectorSet(b *testing.B) {
// 	cx := cvf(b.N)
// 	ca := 0.4 - 0.7i

// 	b.ResetTimer()
// 	for i := 0; i < b.N; i++ {
// 		cx.Set(i, ca)
// 	}
// }
