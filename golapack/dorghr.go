package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorghr generates a real orthogonal matrix Q which is defined as the
// product of IHI-ILO elementary reflectors of order N, as returned by
// DGEHRD:
//
// Q = H(ilo) H(ilo+1) . . . H(ihi-1).
func Dorghr(n, ilo, ihi int, a *mat.Matrix, tau, work *mat.Vector, lwork int) (err error) {
	var lquery bool
	var one, zero float64
	var i, j, lwkopt, nb, nh int

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	nh = ihi - ilo
	lquery = (lwork == -1)
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ilo < 1 || ilo > max(1, n) {
		err = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=%v, n=%v", ilo, n)
	} else if ihi < min(ilo, n) || ihi > n {
		err = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ihi=%v, ilo=%v, n=%v", ihi, ilo, n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if lwork < max(1, nh) && !lquery {
		err = fmt.Errorf("lwork < max(1, nh) && !lquery: lwork=%v, nh=%v, lquery=%v", lwork, nh, lquery)
	}

	if err == nil {
		nb = Ilaenv(1, "Dorgqr", []byte{' '}, nh, nh, nh, -1)
		lwkopt = max(1, nh) * nb
		work.Set(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Dorghr", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		work.Set(0, 1)
		return
	}

	//     Shift the vectors which define the elementary reflectors one
	//     column to the right, and set the first ilo and the last n-ihi
	//     rows and columns to those of the unit matrix
	for j = ihi; j >= ilo+1; j-- {
		for i = 1; i <= j-1; i++ {
			a.Set(i-1, j-1, zero)
		}
		for i = j + 1; i <= ihi; i++ {
			a.Set(i-1, j-1, a.Get(i-1, j-1-1))
		}
		for i = ihi + 1; i <= n; i++ {
			a.Set(i-1, j-1, zero)
		}
	}
	for j = 1; j <= ilo; j++ {
		for i = 1; i <= n; i++ {
			a.Set(i-1, j-1, zero)
		}
		a.Set(j-1, j-1, one)
	}
	for j = ihi + 1; j <= n; j++ {
		for i = 1; i <= n; i++ {
			a.Set(i-1, j-1, zero)
		}
		a.Set(j-1, j-1, one)
	}

	if nh > 0 {
		//        Generate Q(ilo+1:ihi,ilo+1:ihi)
		if err = Dorgqr(nh, nh, nh, a.Off(ilo, ilo), tau.Off(ilo-1), work, lwork); err != nil {
			panic(err)
		}
	}
	work.Set(0, float64(lwkopt))

	return
}
