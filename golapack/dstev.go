package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dstev computes all eigenvalues and, optionally, eigenvectors of a
// real symmetric tridiagonal matrix A.
func Dstev(jobz byte, n int, d, e *mat.Vector, z *mat.Matrix, work *mat.Vector) (info int, err error) {
	var wantz bool
	var bignum, eps, one, rmax, rmin, safmin, sigma, smlnum, tnrm, zero float64
	var imax, iscale int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	wantz = jobz == 'V'

	if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): joz='%c'", jobz)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if z.Rows < 1 || (wantz && z.Rows < n) {
		err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n): jobz='%c', z.Rows=%v, n=%v", jobz, z.Rows, n)
	}

	if err != nil {
		gltest.Xerbla2("Dstev", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
		if wantz {
			z.Set(0, 0, one)
		}
		return
	}

	//     Get machine constants.
	safmin = Dlamch(SafeMinimum)
	eps = Dlamch(Precision)
	smlnum = safmin / eps
	bignum = one / smlnum
	rmin = math.Sqrt(smlnum)
	rmax = math.Sqrt(bignum)

	//     Scale matrix to allowable range, if necessary.
	iscale = 0
	tnrm = Dlanst('M', n, d, e)
	if tnrm > zero && tnrm < rmin {
		iscale = 1
		sigma = rmin / tnrm
	} else if tnrm > rmax {
		iscale = 1
		sigma = rmax / tnrm
	}
	if iscale == 1 {
		goblas.Dscal(n, sigma, d.Off(0, 1))
		goblas.Dscal(n-1, sigma, e.Off(0, 1))
	}

	//     For eigenvalues only, call DSTERF.  For eigenvalues and
	//     eigenvectors, call DSTEQR.
	if !wantz {
		if info, err = Dsterf(n, d, e); err != nil {
			panic(err)
		}
	} else {
		if info, err = Dsteqr('I', n, d, e, z, work); err != nil {
			panic(err)
		}
	}

	//     If matrix was scaled, then rescale eigenvalues appropriately.
	if iscale == 1 {
		if info == 0 {
			imax = n
		} else {
			imax = info - 1
		}
		goblas.Dscal(imax, one/sigma, d.Off(0, 1))
	}

	return
}
