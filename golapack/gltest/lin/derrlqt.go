package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Derlqt tests the error exits for the DOUBLE PRECISION routines
// that use the LQT decomposition of a general matrix.
func Derrlqt(path []byte, _t *testing.T) {
	var i, info, j, nmax int
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 2

	a := mf(2, 2, opts)
	c := mf(2, 2, opts)
	t := mf(2, 2, opts)
	w := vf(2)

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			c.Set(i-1, j-1, 1./float64(i+j))
			t.Set(i-1, j-1, 1./float64(i+j))
		}
		w.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for LQT factorization
	//
	//     DGELQT
	*srnamt = "DGELQT"
	*infot = 1
	golapack.Dgelqt(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGELQT", &info, lerr, ok, _t)
	*infot = 2
	golapack.Dgelqt(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGELQT", &info, lerr, ok, _t)
	*infot = 3
	golapack.Dgelqt(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGELQT", &info, lerr, ok, _t)
	*infot = 5
	golapack.Dgelqt(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGELQT", &info, lerr, ok, _t)
	*infot = 7
	golapack.Dgelqt(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGELQT", &info, lerr, ok, _t)

	//     DGELQT3
	*srnamt = "DGELQT3"
	*infot = 1
	golapack.Dgelqt3(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQT3", &info, lerr, ok, _t)
	*infot = 2
	golapack.Dgelqt3(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQT3", &info, lerr, ok, _t)
	*infot = 4
	golapack.Dgelqt3(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQT3", &info, lerr, ok, _t)
	*infot = 6
	golapack.Dgelqt3(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGELQT3", &info, lerr, ok, _t)

	//     DGEMLQT
	*srnamt = "DGEMLQT"
	*infot = 1
	golapack.Dgemlqt('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMLQT", &info, lerr, ok, _t)
	*infot = 2
	golapack.Dgemlqt('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMLQT", &info, lerr, ok, _t)
	*infot = 3
	golapack.Dgemlqt('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMLQT", &info, lerr, ok, _t)
	*infot = 4
	golapack.Dgemlqt('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMLQT", &info, lerr, ok, _t)
	*infot = 5
	golapack.Dgemlqt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMLQT", &info, lerr, ok, _t)
	*infot = 5
	golapack.Dgemlqt('R', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMLQT", &info, lerr, ok, _t)
	*infot = 6
	golapack.Dgemlqt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMLQT", &info, lerr, ok, _t)
	*infot = 8
	golapack.Dgemlqt('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMLQT", &info, lerr, ok, _t)
	*infot = 8
	golapack.Dgemlqt('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMLQT", &info, lerr, ok, _t)
	*infot = 10
	golapack.Dgemlqt('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMLQT", &info, lerr, ok, _t)
	*infot = 12
	golapack.Dgemlqt('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), c, func() *int { y := 0; return &y }(), w, &info)
	Chkxer("DGEMLQT", &info, lerr, ok, _t)

	//     Print a summary line.
	Alaesm(path, ok)
}
