package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Zerlqt tests the error exits for the COMPLEX routines
// that use the LQT decomposition of a general matrix.
func Zerrlqt(path []byte, _t *testing.T) {
	var i, info, j, nmax int

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
			a.Set(i-1, j-1, 1./complex(float64(i+j), 0))
			c.Set(i-1, j-1, 1./complex(float64(i+j), 0))
			t.Set(i-1, j-1, 1./complex(float64(i+j), 0))
		}
		w.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for LQT factorization
	//
	//     ZGELQT
	*srnamt = "ZGELQT"
	(*infot) = 1
	golapack.Zgelqt(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGELQT", &info, lerr, ok, _t)
	(*infot) = 2
	golapack.Zgelqt(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGELQT", &info, lerr, ok, _t)
	(*infot) = 3
	golapack.Zgelqt(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGELQT", &info, lerr, ok, _t)
	(*infot) = 5
	golapack.Zgelqt(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGELQT", &info, lerr, ok, _t)
	(*infot) = 7
	golapack.Zgelqt(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGELQT", &info, lerr, ok, _t)

	//     ZGELQT3
	*srnamt = "ZGELQT3"
	(*infot) = 1
	golapack.Zgelqt3(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQT3", &info, lerr, ok, _t)
	(*infot) = 2
	golapack.Zgelqt3(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQT3", &info, lerr, ok, _t)
	(*infot) = 4
	golapack.Zgelqt3(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQT3", &info, lerr, ok, _t)
	(*infot) = 6
	golapack.Zgelqt3(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZGELQT3", &info, lerr, ok, _t)

	//     ZGEMLQT
	*srnamt = "ZGEMLQT"
	(*infot) = 1
	golapack.Zgemlqt('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMLQT", &info, lerr, ok, _t)
	(*infot) = 2
	golapack.Zgemlqt('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMLQT", &info, lerr, ok, _t)
	(*infot) = 3
	golapack.Zgemlqt('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMLQT", &info, lerr, ok, _t)
	(*infot) = 4
	golapack.Zgemlqt('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMLQT", &info, lerr, ok, _t)
	(*infot) = 5
	golapack.Zgemlqt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMLQT", &info, lerr, ok, _t)
	(*infot) = 5
	golapack.Zgemlqt('R', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMLQT", &info, lerr, ok, _t)
	(*infot) = 6
	golapack.Zgemlqt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMLQT", &info, lerr, ok, _t)
	(*infot) = 8
	golapack.Zgemlqt('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMLQT", &info, lerr, ok, _t)
	(*infot) = 8
	golapack.Zgemlqt('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMLQT", &info, lerr, ok, _t)
	(*infot) = 10
	golapack.Zgemlqt('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZGEMLQT", &info, lerr, ok, _t)
	(*infot) = 12
	golapack.Zgemlqt('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 0; return &y }(), w, &info)
	Chkxer("ZGEMLQT", &info, lerr, ok, _t)

	//     Print a summary line.
	Alaesm(path, ok)
}
