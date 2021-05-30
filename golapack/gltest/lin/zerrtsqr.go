package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Zerrtsqr tests the error exits for the ZOUBLE PRECISION routines
// that use the TSQR decomposition of a general matrix.
func Zerrtsqr(path []byte, _t *testing.T) {
	var i, info, j, nmax int

	tau := cvf(4)
	w := cvf(2)
	a := cmf(2, 2, opts)
	c := cmf(2, 2, opts)
	t := cmf(2, 2, opts)

	nmax = 2
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.SetRe(i-1, j-1, 1./float64(i+j))
			c.SetRe(i-1, j-1, 1./float64(i+j))
			t.SetRe(i-1, j-1, 1./float64(i+j))
		}
		w.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for TS factorization
	//
	//     ZGEQR
	*srnamt = "ZGEQR"
	*infot = 1
	golapack.Zgeqr(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQR", &info, lerr, ok, _t)
	*infot = 2
	golapack.Zgeqr(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQR", &info, lerr, ok, _t)
	*infot = 4
	golapack.Zgeqr(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQR", &info, lerr, ok, _t)
	*infot = 6
	golapack.Zgeqr(func() *int { y := 3; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 3; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEQR", &info, lerr, ok, _t)
	*infot = 8
	golapack.Zgeqr(func() *int { y := 3; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 3; return &y }(), tau, func() *int { y := 8; return &y }(), w, func() *int { y := 0; return &y }(), &info)
	Chkxer("ZGEQR", &info, lerr, ok, _t)

	//     ZGEMQR
	tau.Set(0, 1)
	tau.Set(1, 1)
	*srnamt = "ZGEMQR"
	// nb = 1
	*infot = 1
	golapack.Zgemqr('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMQR", &info, lerr, ok, _t)
	*infot = 2
	golapack.Zgemqr('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMQR", &info, lerr, ok, _t)
	*infot = 3
	golapack.Zgemqr('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMQR", &info, lerr, ok, _t)
	*infot = 4
	golapack.Zgemqr('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMQR", &info, lerr, ok, _t)
	*infot = 5
	golapack.Zgemqr('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMQR", &info, lerr, ok, _t)
	*infot = 5
	golapack.Zgemqr('R', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMQR", &info, lerr, ok, _t)
	*infot = 7
	golapack.Zgemqr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 0; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMQR", &info, lerr, ok, _t)
	*infot = 9
	golapack.Zgemqr('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tau, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMQR", &info, lerr, ok, _t)
	*infot = 9
	golapack.Zgemqr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tau, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMQR", &info, lerr, ok, _t)
	*infot = 11
	golapack.Zgemqr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tau, func() *int { y := 6; return &y }(), c, func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMQR", &info, lerr, ok, _t)
	*infot = 13
	golapack.Zgemqr('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tau, func() *int { y := 6; return &y }(), c, func() *int { y := 2; return &y }(), w, func() *int { y := 0; return &y }(), &info)
	Chkxer("ZGEMQR", &info, lerr, ok, _t)

	//     ZGELQ
	*srnamt = "ZGELQ"
	*infot = 1
	golapack.Zgelq(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQ", &info, lerr, ok, _t)
	*infot = 2
	golapack.Zgelq(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQ", &info, lerr, ok, _t)
	*infot = 4
	golapack.Zgelq(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQ", &info, lerr, ok, _t)
	*infot = 6
	golapack.Zgelq(func() *int { y := 2; return &y }(), func() *int { y := 3; return &y }(), a, func() *int { y := 3; return &y }(), tau, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQ", &info, lerr, ok, _t)
	*infot = 8
	golapack.Zgelq(func() *int { y := 2; return &y }(), func() *int { y := 3; return &y }(), a, func() *int { y := 3; return &y }(), tau, func() *int { y := 8; return &y }(), w, func() *int { y := 0; return &y }(), &info)
	Chkxer("ZGELQ", &info, lerr, ok, _t)

	//     ZGEMLQ
	tau.Set(0, 1)
	tau.Set(1, 1)
	*srnamt = "ZGEMLQ"
	// nb = 1
	*infot = 1
	golapack.Zgemlq('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMLQ", &info, lerr, ok, _t)
	*infot = 2
	golapack.Zgemlq('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMLQ", &info, lerr, ok, _t)
	*infot = 3
	golapack.Zgemlq('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMLQ", &info, lerr, ok, _t)
	*infot = 4
	golapack.Zgemlq('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMLQ", &info, lerr, ok, _t)
	*infot = 5
	golapack.Zgemlq('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMLQ", &info, lerr, ok, _t)
	*infot = 5
	golapack.Zgemlq('R', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMLQ", &info, lerr, ok, _t)
	*infot = 7
	golapack.Zgemlq('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 0; return &y }(), tau, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMLQ", &info, lerr, ok, _t)
	*infot = 9
	golapack.Zgemlq('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMLQ", &info, lerr, ok, _t)
	*infot = 9
	golapack.Zgemlq('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMLQ", &info, lerr, ok, _t)
	*infot = 11
	golapack.Zgemlq('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), tau, func() *int { y := 6; return &y }(), c, func() *int { y := 0; return &y }(), w, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGEMLQ", &info, lerr, ok, _t)
	*infot = 13
	golapack.Zgemlq('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), tau, func() *int { y := 6; return &y }(), c, func() *int { y := 2; return &y }(), w, func() *int { y := 0; return &y }(), &info)
	Chkxer("ZGEMLQ", &info, lerr, ok, _t)

	//     Print a summary line.
	Alaesm(path, ok)
}
