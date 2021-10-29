package matgen

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dlatm6 generates test matrices for the generalized eigenvalue
// problem, their corresponding right and left eigenvector matrices,
// and also reciprocal condition numbers for all eigenvalues and
// the reciprocal condition numbers of eigenvectors corresponding to
// the 1th and 5th eigenvalues.
//
// Test Matrices
// =============
//
// Two kinds of test matrix pairs
//
//       (A, B) = inverse(YH) * (Da, Db) * inverse(X)
//
// are used in the tests:
//
// Type 1:
//    Da = 1+a   0    0    0    0    Db = 1   0   0   0   0
//          0   2+a   0    0    0         0   1   0   0   0
//          0    0   3+a   0    0         0   0   1   0   0
//          0    0    0   4+a   0         0   0   0   1   0
//          0    0    0    0   5+a ,      0   0   0   0   1 , and
//
// Type 2:
//    Da =  1   -1    0    0    0    Db = 1   0   0   0   0
//          1    1    0    0    0         0   1   0   0   0
//          0    0    1    0    0         0   0   1   0   0
//          0    0    0   1+a  1+b        0   0   0   1   0
//          0    0    0  -1-b  1+a ,      0   0   0   0   1 .
//
// In both cases the same inverse(YH) and inverse(X) are used to compute
// (A, B), giving the exact eigenvectors to (A,B) as (YH, X):
//
// YH:  =  1    0   -y    y   -y    X =  1   0  -x  -x   x
//         0    1   -y    y   -y         0   1   x  -x  -x
//         0    0    1    0    0         0   0   1   0   0
//         0    0    0    1    0         0   0   0   1   0
//         0    0    0    0    1,        0   0   0   0   1 ,
//
// where a, b, x and y will have all values independently of each other.
func Dlatm6(_type, n int, a, b, x, y *mat.Matrix, alpha, beta, wx, wy float64, s, dif *mat.Vector) {
	var one, three, two, zero float64
	var i, j int
	var err error

	work := vf(100)
	z := mf(12, 12, opts)

	zero = 0.0
	one = 1.0
	two = 2.0
	three = 3.0

	//     Generate test problem ...
	//     (Da, Db) ...
	for i = 1; i <= n; i++ {
		for j = 1; j <= n; j++ {

			if i == j {
				a.Set(i-1, i-1, float64(i)+alpha)
				b.Set(i-1, i-1, one)
			} else {
				a.Set(i-1, j-1, zero)
				b.Set(i-1, j-1, zero)
			}

		}
	}

	//     Form X and Y
	golapack.Dlacpy(Full, n, n, b, y)
	y.Set(2, 0, -wy)
	y.Set(3, 0, wy)
	y.Set(4, 0, -wy)
	y.Set(2, 1, -wy)
	y.Set(3, 1, wy)
	y.Set(4, 1, -wy)

	golapack.Dlacpy(Full, n, n, b, x)
	x.Set(0, 2, -wx)
	x.Set(0, 3, -wx)
	x.Set(0, 4, wx)
	x.Set(1, 2, wx)
	x.Set(1, 3, -wx)
	x.Set(1, 4, -wx)

	//     Form (A, B)
	b.Set(0, 2, wx+wy)
	b.Set(1, 2, -wx+wy)
	b.Set(0, 3, wx-wy)
	b.Set(1, 3, wx-wy)
	b.Set(0, 4, -wx+wy)
	b.Set(1, 4, wx+wy)
	if _type == 1 {
		a.Set(0, 2, wx*a.Get(0, 0)+wy*a.Get(2, 2))
		a.Set(1, 2, -wx*a.Get(1, 1)+wy*a.Get(2, 2))
		a.Set(0, 3, wx*a.Get(0, 0)-wy*a.Get(3, 3))
		a.Set(1, 3, wx*a.Get(1, 1)-wy*a.Get(3, 3))
		a.Set(0, 4, -wx*a.Get(0, 0)+wy*a.Get(4, 4))
		a.Set(1, 4, wx*a.Get(1, 1)+wy*a.Get(4, 4))
	} else if _type == 2 {
		a.Set(0, 2, two*wx+wy)
		a.Set(1, 2, wy)
		a.Set(0, 3, -wy*(two+alpha+beta))
		a.Set(1, 3, two*wx-wy*(two+alpha+beta))
		a.Set(0, 4, -two*wx+wy*(alpha-beta))
		a.Set(1, 4, wy*(alpha-beta))
		a.Set(0, 0, one)
		a.Set(0, 1, -one)
		a.Set(1, 0, one)
		a.Set(1, 1, a.Get(0, 0))
		a.Set(2, 2, one)
		a.Set(3, 3, one+alpha)
		a.Set(3, 4, one+beta)
		a.Set(4, 3, -a.Get(3, 4))
		a.Set(4, 4, a.Get(3, 3))
	}

	//     Compute condition numbers
	if _type == 1 {

		s.Set(0, one/math.Sqrt((one+three*wy*wy)/(one+a.Get(0, 0)*a.Get(0, 0))))
		s.Set(1, one/math.Sqrt((one+three*wy*wy)/(one+a.Get(1, 1)*a.Get(1, 1))))
		s.Set(2, one/math.Sqrt((one+two*wx*wx)/(one+a.Get(2, 2)*a.Get(2, 2))))
		s.Set(3, one/math.Sqrt((one+two*wx*wx)/(one+a.Get(3, 3)*a.Get(3, 3))))
		s.Set(4, one/math.Sqrt((one+two*wx*wx)/(one+a.Get(4, 4)*a.Get(4, 4))))

		Dlakf2(1, 4, a, a.Off(1, 1), b, b.Off(1, 1), z)
		if _, err = golapack.Dgesvd('N', 'N', 8, 8, z, work, work.MatrixOff(8, 1, opts), work.MatrixOff(9, 1, opts), work.Off(10), 40); err != nil {
			panic(err)
		}
		dif.Set(0, work.Get(7))

		Dlakf2(4, 1, a, a.Off(4, 4), b, b.Off(4, 4), z)
		if _, err = golapack.Dgesvd('N', 'N', 8, 8, z, work, work.MatrixOff(8, 1, opts), work.MatrixOff(9, 1, opts), work.Off(10), 40); err != nil {
			panic(err)
		}
		dif.Set(4, work.Get(7))

	} else if _type == 2 {

		s.Set(0, one/math.Sqrt(one/three+wy*wy))
		s.Set(1, s.Get(0))
		s.Set(2, one/math.Sqrt(one/two+wx*wx))
		s.Set(3, one/math.Sqrt((one+two*wx*wx)/(one+(one+alpha)*(one+alpha)+(one+beta)*(one+beta))))
		s.Set(4, s.Get(3))

		Dlakf2(2, 3, a, a.Off(2, 2), b, b.Off(2, 2), z)
		if _, err = golapack.Dgesvd('N', 'N', 12, 12, z, work, work.MatrixOff(12, 1, opts), work.MatrixOff(13, 1, opts), work.Off(14), 60); err != nil {
			panic(err)
		}
		dif.Set(0, work.Get(11))

		Dlakf2(3, 2, a, a.Off(3, 3), b, b.Off(3, 3), z)
		if _, err = golapack.Dgesvd('N', 'N', 12, 12, z, work, work.MatrixOff(12, 1, opts), work.MatrixOff(13, 1, opts), work.Off(14), 60); err != nil {
			panic(err)
		}
		dif.Set(4, work.Get(11))

	}
}
