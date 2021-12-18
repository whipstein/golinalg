package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
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
func Dlasq1(n int, d, e, work *mat.Vector) (info int, err error) {
	var eps, safmin, scale, sigmn, sigmx, zero float64
	var i int

	zero = 0.0

	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
		gltest.Xerbla2("Dlasq1", err)
		return
	} else if n == 0 {
		return
	} else if n == 1 {
		d.Set(0, math.Abs(d.Get(0)))
		return
	} else if n == 2 {
		sigmn, sigmx = Dlas2(d.Get(0), e.Get(0), d.Get(1))
		d.Set(0, sigmx)
		d.Set(1, sigmn)
		return
	}

	//     Estimate the largest singular value.
	sigmx = zero
	for i = 1; i <= n-1; i++ {
		d.Set(i-1, math.Abs(d.Get(i-1)))
		sigmx = math.Max(sigmx, math.Abs(e.Get(i-1)))
	}
	d.Set(n-1, math.Abs(d.Get(n-1)))

	//     Early return if SIGMX is zero (matrix is already diagonal).
	if sigmx == zero {
		if err = Dlasrt('D', n, d); err != nil {
			panic(err)
		}
		return
	}

	for i = 1; i <= n; i++ {
		sigmx = math.Max(sigmx, d.Get(i-1))
	}

	//     Copy D and E into WORK (in the Z format) and scale (squaring the
	//     input data makes scaling by a power of the radix pointless).
	eps = Dlamch(Precision)
	safmin = Dlamch(SafeMinimum)
	scale = math.Sqrt(eps / safmin)
	work.Copy(n, d, 1, 2)
	work.Off(1).Copy(n-1, e, 1, 2)
	if err = Dlascl('G', 0, 0, sigmx, scale, 2*n-1, 1, work.Matrix(2*n-1, opts)); err != nil {
		panic(err)
	}

	//     Compute the q's and e's.
	for i = 1; i <= 2*n-1; i++ {
		work.Set(i-1, math.Pow(work.Get(i-1), 2))
	}
	work.Set(2*n-1, zero)

	if info, err = Dlasq2(n, work); err != nil {
		panic(err)
	}

	if info == 0 {
		for i = 1; i <= n; i++ {
			d.Set(i-1, math.Sqrt(work.Get(i-1)))
		}
		if err = Dlascl('G', 0, 0, scale, sigmx, n, 1, d.Matrix(n, opts)); err != nil {
			panic(err)
		}
	} else if info == 2 {
		//     Maximum number of iterations exceeded.  Move data from WORK
		//     into D and E so the calling subroutine can try to finish
		for i = 1; i <= n; i++ {
			d.Set(i-1, math.Sqrt(work.Get(2*i-1-1)))
			e.Set(i-1, math.Sqrt(work.Get(2*i-1)))
		}
		if err = Dlascl('G', 0, 0, scale, sigmx, n, 1, d.Matrix(n, opts)); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, scale, sigmx, n, 1, e.Matrix(n, opts)); err != nil {
			panic(err)
		}
	}

	return
}
