package matgen

import (
	"golinalg/golapack"
	"golinalg/mat"
	"math"
	"math/cmplx"
)

// Zlatm6 generates test matrices for the generalized eigenvalue
// problem, their corresponding right and left eigenvector matrices,
// and also reciprocal condition numbers for all eigenvalues and
// the reciprocal condition numbers of eigenvectors corresponding to
// the 1th and 5th eigenvalues.
//
// Test Matrices
// =============
//
// Two kinds of test matrix pairs
//          (A, B) = inverse(YH) * (Da, Db) * inverse(X)
// are used in the tests:
//
// Type 1:
//    Da = 1+a   0    0    0    0    Db = 1   0   0   0   0
//          0   2+a   0    0    0         0   1   0   0   0
//          0    0   3+a   0    0         0   0   1   0   0
//          0    0    0   4+a   0         0   0   0   1   0
//          0    0    0    0   5+a ,      0   0   0   0   1
// and Type 2:
//    Da = 1+i   0    0       0       0    Db = 1   0   0   0   0
//          0   1-i   0       0       0         0   1   0   0   0
//          0    0    1       0       0         0   0   1   0   0
//          0    0    0 (1+a)+(1+b)i  0         0   0   0   1   0
//          0    0    0       0 (1+a)-(1+b)i,   0   0   0   0   1 .
//
// In both cases the same inverse(YH) and inverse(X) are used to compute
// (A, B), giving the exact eigenvectors to (A,B) as (YH, X):
//
// YH:  =  1    0   -y    y   -y    X =  1   0  -x  -x   x
//         0    1   -y    y   -y         0   1   x  -x  -x
//         0    0    1    0    0         0   0   1   0   0
//         0    0    0    1    0         0   0   0   1   0
//         0    0    0    0    1,        0   0   0   0   1 , where
//
// a, b, x and y will have all values independently of each other.
func Zlatm6(_type, n *int, a *mat.CMatrix, lda *int, b, x *mat.CMatrix, ldx *int, y *mat.CMatrix, ldy *int, alpha, beta, wx, wy *complex128, s, dif *mat.Vector) {
	var one, zero complex128
	var rone, three, two float64
	var i, info, j int
	work := cvf(26)
	rwork := vf(50)
	z := cmf(8, 8, opts)

	rone = 1.0
	two = 2.0
	three = 3.0
	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Generate test problem ...
	//     (Da, Db) ...
	for i = 1; i <= (*n); i++ {
		for j = 1; j <= (*n); j++ {

			if i == j {
				a.Set(i-1, i-1, toCmplx(float64(i))+(*alpha))
				b.Set(i-1, i-1, one)
			} else {
				a.Set(i-1, j-1, zero)
				b.Set(i-1, j-1, zero)
			}

		}
	}
	if (*_type) == 2 {
		a.Set(0, 0, complex(rone, rone))
		a.Set(1, 1, a.GetConj(0, 0))
		a.Set(2, 2, one)
		a.Set(3, 3, complex(real(one+(*alpha)), real(one+(*beta))))
		a.Set(4, 4, a.GetConj(3, 3))
	}

	//     Form X and Y
	golapack.Zlacpy('F', n, n, b, lda, y, ldy)
	y.Set(2, 0, -cmplx.Conj(*wy))
	y.Set(3, 0, cmplx.Conj(*wy))
	y.Set(4, 0, -cmplx.Conj(*wy))
	y.Set(2, 1, -cmplx.Conj(*wy))
	y.Set(3, 1, cmplx.Conj(*wy))
	y.Set(4, 1, -cmplx.Conj(*wy))

	golapack.Zlacpy('F', n, n, b, lda, x, ldx)
	x.Set(0, 2, -(*wx))
	x.Set(0, 3, -(*wx))
	x.Set(0, 4, (*wx))
	x.Set(1, 2, (*wx))
	x.Set(1, 3, -(*wx))
	x.Set(1, 4, -(*wx))

	//     Form (A, B)
	b.Set(0, 2, (*wx)+(*wy))
	b.Set(1, 2, -(*wx)+(*wy))
	b.Set(0, 3, (*wx)-(*wy))
	b.Set(1, 3, (*wx)-(*wy))
	b.Set(0, 4, -(*wx)+(*wy))
	b.Set(1, 4, (*wx)+(*wy))
	a.Set(0, 2, (*wx)*a.Get(0, 0)+(*wy)*a.Get(2, 2))
	a.Set(1, 2, -(*wx)*a.Get(1, 1)+(*wy)*a.Get(2, 2))
	a.Set(0, 3, (*wx)*a.Get(0, 0)-(*wy)*a.Get(3, 3))
	a.Set(1, 3, (*wx)*a.Get(1, 1)-(*wy)*a.Get(3, 3))
	a.Set(0, 4, -(*wx)*a.Get(0, 0)+(*wy)*a.Get(4, 4))
	a.Set(1, 4, (*wx)*a.Get(1, 1)+(*wy)*a.Get(4, 4))

	//     Compute condition numbers
	s.Set(0, rone/math.Sqrt((rone+three*cmplx.Abs(*wy)*cmplx.Abs(*wy))/(rone+a.GetMag(0, 0)*a.GetMag(0, 0))))
	s.Set(1, rone/math.Sqrt((rone+three*cmplx.Abs(*wy)*cmplx.Abs(*wy))/(rone+a.GetMag(1, 1)*a.GetMag(1, 1))))
	s.Set(2, rone/math.Sqrt((rone+two*cmplx.Abs(*wx)*cmplx.Abs(*wx))/(rone+a.GetMag(2, 2)*a.GetMag(2, 2))))
	s.Set(3, rone/math.Sqrt((rone+two*cmplx.Abs(*wx)*cmplx.Abs(*wx))/(rone+a.GetMag(3, 3)*a.GetMag(3, 3))))
	s.Set(4, rone/math.Sqrt((rone+two*cmplx.Abs(*wx)*cmplx.Abs(*wx))/(rone+a.GetMag(4, 4)*a.GetMag(4, 4))))

	Zlakf2(func() *int { y := 1; return &y }(), func() *int { y := 4; return &y }(), a, lda, a.Off(1, 1), b, b.Off(1, 1), z, func() *int { y := 8; return &y }())
	golapack.Zgesvd('N', 'N', func() *int { y := 8; return &y }(), func() *int { y := 8; return &y }(), z, func() *int { y := 8; return &y }(), rwork, work.CMatrix(1, opts), func() *int { y := 1; return &y }(), work.CMatrixOff(1, 1, opts), func() *int { y := 1; return &y }(), work.Off(2), func() *int { y := 24; return &y }(), rwork.Off(8), &info)
	dif.Set(0, rwork.Get(7))

	Zlakf2(func() *int { y := 4; return &y }(), func() *int { y := 1; return &y }(), a, lda, a.Off(4, 4), b, b.Off(4, 4), z, func() *int { y := 8; return &y }())
	golapack.Zgesvd('N', 'N', func() *int { y := 8; return &y }(), func() *int { y := 8; return &y }(), z, func() *int { y := 8; return &y }(), rwork, work.CMatrix(1, opts), func() *int { y := 1; return &y }(), work.CMatrixOff(1, 1, opts), func() *int { y := 1; return &y }(), work.Off(2), func() *int { y := 24; return &y }(), rwork.Off(8), &info)
	dif.Set(4, rwork.Get(7))
}
