package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dlasq1 computes the singular values of a real N-by-N bidiagonal
// matrix with diagonal D and off-diagonal E. The singular values
// are computed to high relative accuracy, in the absence of
// denormalization, underflow and overflow. The algorithm was first
// presented in
//
// "Accurate singular values and differential qd algorithms" by K. V.
// Fernando and B. N. Parlett, Numer. Math., Vol-67, No. 2, pp. 191-230,
// 1994,
//
// and the present implementation is described in "An implementation of
// the dqds Algorithm (Positive Case)", LAPACK Working Note.
func Dlasq1(n *int, d, e, work *mat.Vector, info *int) {
	var eps, safmin, scale, sigmn, sigmx, zero float64
	var i, iinfo int

	zero = 0.0

	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
		gltest.Xerbla([]byte("DLASQ1"), -(*info))
		return
	} else if (*n) == 0 {
		return
	} else if (*n) == 1 {
		d.Set(0, math.Abs(d.Get(0)))
		return
	} else if (*n) == 2 {
		Dlas2(d.GetPtr(0), e.GetPtr(0), d.GetPtr(1), &sigmn, &sigmx)
		d.Set(0, sigmx)
		d.Set(1, sigmn)
		return
	}

	//     Estimate the largest singular value.
	sigmx = zero
	for i = 1; i <= (*n)-1; i++ {
		d.Set(i-1, math.Abs(d.Get(i-1)))
		sigmx = maxf64(sigmx, math.Abs(e.Get(i-1)))
	}
	d.Set((*n)-1, math.Abs(d.Get((*n)-1)))

	//     Early return if SIGMX is zero (matrix is already diagonal).
	if sigmx == zero {
		Dlasrt('D', n, d, &iinfo)
		return
	}

	for i = 1; i <= (*n); i++ {
		sigmx = maxf64(sigmx, d.Get(i-1))
	}

	//     Copy D and E into WORK (in the Z format) and scale (squaring the
	//     input data makes scaling by a power of the radix pointless).
	eps = Dlamch(Precision)
	safmin = Dlamch(SafeMinimum)
	scale = math.Sqrt(eps / safmin)
	goblas.Dcopy(n, d, toPtr(1), work, toPtr(2))
	goblas.Dcopy(toPtr((*n)-1), e, toPtr(1), work.Off(1), toPtr(2))
	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &sigmx, &scale, toPtr(2*(*n)-1), func() *int { y := 1; return &y }(), work.Matrix(2*(*n)-1, opts), toPtr(2*(*n)-1), &iinfo)

	//     Compute the q's and e's.
	for i = 1; i <= 2*(*n)-1; i++ {
		work.Set(i-1, math.Pow(work.Get(i-1), 2))
	}
	work.Set(2*(*n)-1, zero)

	Dlasq2(n, work, info)

	if (*info) == 0 {
		for i = 1; i <= (*n); i++ {
			d.Set(i-1, math.Sqrt(work.Get(i-1)))
		}
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &scale, &sigmx, n, func() *int { y := 1; return &y }(), d.Matrix(*n, opts), n, &iinfo)
	} else if (*info) == 2 {
		//     Maximum number of iterations exceeded.  Move data from WORK
		//     into D and E so the calling subroutine can try to finish
		for i = 1; i <= (*n); i++ {
			d.Set(i-1, math.Sqrt(work.Get(2*i-1-1)))
			e.Set(i-1, math.Sqrt(work.Get(2*i-1)))
		}
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &scale, &sigmx, n, func() *int { y := 1; return &y }(), d.Matrix(*n, opts), n, &iinfo)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &scale, &sigmx, n, func() *int { y := 1; return &y }(), e.Matrix(*n, opts), n, &iinfo)
	}
}
