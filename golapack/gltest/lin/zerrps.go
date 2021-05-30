package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Zerrps tests the error exits for the COMPLEX routines
// for ZPSTRF.
func Zerrps(path []byte, t *testing.T) {
	var i, info, j, nmax, rank int

	nmax = 4
	rwork := vf(2 * nmax)
	piv := make([]int, 4)
	a := cmf(4, 4, opts)
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.SetRe(i-1, j-1, 1./float64(i+j))

		}
		piv[j-1] = j
		rwork.Set(j-1, 0.)
		rwork.Set(nmax+j-1, 0.)

	}
	*ok = true

	//        Test error exits of the routines that use the Cholesky
	//        decomposition of an Hermitian positive semidefinite matrix.
	//
	//        ZPSTRF
	*srnamt = "ZPSTRF"
	*infot = 1
	golapack.Zpstrf('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &piv, &rank, toPtrf64(-1.), rwork, &info)
	Chkxer("ZPSTRF", &info, lerr, ok, t)
	*infot = 2
	golapack.Zpstrf('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &piv, &rank, toPtrf64(-1.), rwork, &info)
	Chkxer("ZPSTRF", &info, lerr, ok, t)
	*infot = 4
	golapack.Zpstrf('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &piv, &rank, toPtrf64(-1.), rwork, &info)
	Chkxer("ZPSTRF", &info, lerr, ok, t)

	//        ZPSTF2
	*srnamt = "ZPSTF2"
	*infot = 1
	golapack.Zpstf2('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &piv, &rank, toPtrf64(-1.), rwork, &info)
	Chkxer("ZPSTF2", &info, lerr, ok, t)
	*infot = 2
	golapack.Zpstf2('U', toPtr(-1), a, func() *int { y := 1; return &y }(), &piv, &rank, toPtrf64(-1.), rwork, &info)
	Chkxer("ZPSTF2", &info, lerr, ok, t)
	*infot = 4
	golapack.Zpstf2('U', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &piv, &rank, toPtrf64(-1.), rwork, &info)
	Chkxer("ZPSTF2", &info, lerr, ok, t)

	//     Print a summary line.
	Alaesm(path, ok)
}
