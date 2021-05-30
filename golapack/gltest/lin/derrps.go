package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrps tests the error exits for the DOUBLE PRECISION routines
// for DPSTRF.
func Derrps(path []byte, t *testing.T) {
	var i, info, j, nmax, rank int
	piv := make([]int, 4)
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 4

	a := mf(4, 4, opts)
	work := vf(2 * nmax)

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))

		}
		piv[j-1] = j
		work.Set(j-1, 0.)
		work.Set(nmax+j-1, 0.)

	}
	(*ok) = true

	//        Test error exits of the routines that use the Cholesky
	//        decomposition of a symmetric positive semidefinite matrix.
	//
	//        DPSTRF
	*srnamt = "DPSTRF"
	*infot = 1
	golapack.Dpstrf('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &piv, &rank, func() *float64 { y := -1.; return &y }(), work, &info)
	Chkxer("DPSTRF", &info, lerr, ok, t)
	*infot = 2
	golapack.Dpstrf('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &piv, &rank, func() *float64 { y := -1.; return &y }(), work, &info)
	Chkxer("DPSTRF", &info, lerr, ok, t)
	*infot = 4
	golapack.Dpstrf('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &piv, &rank, func() *float64 { y := -1.; return &y }(), work, &info)
	Chkxer("DPSTRF", &info, lerr, ok, t)

	//        DPSTF2
	*srnamt = "DPSTF2"
	*infot = 1
	golapack.Dpstf2('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &piv, &rank, func() *float64 { y := -1.; return &y }(), work, &info)
	Chkxer("DPSTF2", &info, lerr, ok, t)
	*infot = 2
	golapack.Dpstf2('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &piv, &rank, func() *float64 { y := -1.; return &y }(), work, &info)
	Chkxer("DPSTF2", &info, lerr, ok, t)
	*infot = 4
	golapack.Dpstf2('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &piv, &rank, func() *float64 { y := -1.; return &y }(), work, &info)
	Chkxer("DPSTF2", &info, lerr, ok, t)

	//     Print a summary line.
	Alaesm(path, ok)
}
