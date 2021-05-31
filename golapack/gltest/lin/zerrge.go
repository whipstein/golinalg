package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zerrge tests the error exits for the COMPLEX*16 routines
// for general matrices.
func Zerrge(path []byte, t *testing.T) {
	var anrm, ccond, rcond float64
	_ = ccond
	var i, info, j, nmax int

	nmax = 4
	b := cvf(4)
	w := cvf(2 * nmax)
	x := cvf(4)
	r := vf(4)
	r1 := vf(4)
	r2 := vf(4)
	ip := make([]int, 4)
	a := cmf(4, 4, opts)
	af := cmf(4, 4, opts)

	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt
	c2 := path[1:3]

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, complex(1./float64(i+j), -1./float64(i+j)))
			af.Set(i-1, j-1, complex(1./float64(i+j), -1./float64(i+j)))
		}
		b.Set(j-1, 0.)
		r1.Set(j-1, 0.)
		r2.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.Set(j-1, 0.)
		ip[j-1] = j
	}
	*ok = true

	//     Test error exits of the routines that use the LU decomposition
	//     of a general matrix.
	if string(c2) == "GE" {
		//        ZGETRF
		*srnamt = "ZGETRF"
		*infot = 1
		golapack.Zgetrf(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGETRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgetrf(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGETRF", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgetrf(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGETRF", &info, lerr, ok, t)

		//        ZGETF2
		*srnamt = "ZGETF2"
		*infot = 1
		golapack.Zgetf2(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGETF2", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgetf2(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGETF2", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgetf2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGETF2", &info, lerr, ok, t)

		//        ZGETRI
		*srnamt = "ZGETRI"
		*infot = 1
		golapack.Zgetri(toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGETRI", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgetri(func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &ip, w, func() *int { y := 2; return &y }(), &info)
		Chkxer("ZGETRI", &info, lerr, ok, t)
		*infot = 6
		golapack.Zgetri(func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), &ip, w, func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGETRI", &info, lerr, ok, t)

		//        ZGETRS
		*srnamt = "ZGETRS"
		*infot = 1
		golapack.Zgetrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGETRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgetrs('N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGETRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgetrs('N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGETRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgetrs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZGETRS", &info, lerr, ok, t)
		*infot = 8
		golapack.Zgetrs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGETRS", &info, lerr, ok, t)

		//        ZGERFS
		*srnamt = "ZGERFS"
		*infot = 1
		golapack.Zgerfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGERFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgerfs('N', toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGERFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgerfs('N', func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGERFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgerfs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGERFS", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgerfs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGERFS", &info, lerr, ok, t)
		*infot = 10
		golapack.Zgerfs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGERFS", &info, lerr, ok, t)
		*infot = 12
		golapack.Zgerfs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 2; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGERFS", &info, lerr, ok, t)

		//        ZGECON
		*srnamt = "ZGECON"
		*infot = 1
		golapack.Zgecon('/', func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, r, &info)
		Chkxer("ZGECON", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgecon('1', toPtr(-1), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, r, &info)
		Chkxer("ZGECON", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgecon('1', func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), &anrm, &rcond, w, r, &info)
		Chkxer("ZGECON", &info, lerr, ok, t)

		//        ZGEEQU
		*srnamt = "ZGEEQU"
		*infot = 1
		golapack.Zgeequ(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("ZGEEQU", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgeequ(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("ZGEEQU", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgeequ(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("ZGEEQU", &info, lerr, ok, t)

		//     Test error exits of the routines that use the LU decomposition
		//     of a general band matrix.
	} else if string(c2) == "GB" {
		//        ZGBTRF
		*srnamt = "ZGBTRF"
		*infot = 1
		golapack.Zgbtrf(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGBTRF", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgbtrf(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGBTRF", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgbtrf(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGBTRF", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgbtrf(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGBTRF", &info, lerr, ok, t)
		*infot = 6
		golapack.Zgbtrf(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 3; return &y }(), &ip, &info)
		Chkxer("ZGBTRF", &info, lerr, ok, t)

		//        ZGBTF2
		*srnamt = "ZGBTF2"
		*infot = 1
		golapack.Zgbtf2(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGBTF2", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgbtf2(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGBTF2", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgbtf2(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGBTF2", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgbtf2(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &info)
		Chkxer("ZGBTF2", &info, lerr, ok, t)
		*infot = 6
		golapack.Zgbtf2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 3; return &y }(), &ip, &info)
		Chkxer("ZGBTF2", &info, lerr, ok, t)

		//        ZGBTRS
		*srnamt = "ZGBTRS"
		*infot = 1
		golapack.Zgbtrs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGBTRS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgbtrs('N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGBTRS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgbtrs('N', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGBTRS", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgbtrs('N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGBTRS", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgbtrs('N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGBTRS", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgbtrs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 3; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), &info)
		Chkxer("ZGBTRS", &info, lerr, ok, t)
		*infot = 10
		golapack.Zgbtrs('N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), &info)
		Chkxer("ZGBTRS", &info, lerr, ok, t)

		//        ZGBRFS
		*srnamt = "ZGBRFS"
		*infot = 1
		golapack.Zgbrfs('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGBRFS", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgbrfs('N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGBRFS", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgbrfs('N', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGBRFS", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgbrfs('N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGBRFS", &info, lerr, ok, t)
		*infot = 5
		golapack.Zgbrfs('N', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGBRFS", &info, lerr, ok, t)
		*infot = 7
		golapack.Zgbrfs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), af, func() *int { y := 4; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGBRFS", &info, lerr, ok, t)
		*infot = 9
		golapack.Zgbrfs('N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 3; return &y }(), af, func() *int { y := 3; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGBRFS", &info, lerr, ok, t)
		*infot = 12
		golapack.Zgbrfs('N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(1, opts), func() *int { y := 1; return &y }(), x.CMatrix(2, opts), func() *int { y := 2; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGBRFS", &info, lerr, ok, t)
		*infot = 14
		golapack.Zgbrfs('N', func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), af, func() *int { y := 1; return &y }(), &ip, b.CMatrix(2, opts), func() *int { y := 2; return &y }(), x.CMatrix(1, opts), func() *int { y := 1; return &y }(), r1, r2, w, r, &info)
		Chkxer("ZGBRFS", &info, lerr, ok, t)

		//        ZGBCON
		*srnamt = "ZGBCON"
		*infot = 1
		golapack.Zgbcon('/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, r, &info)
		Chkxer("ZGBCON", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgbcon('1', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, r, &info)
		Chkxer("ZGBCON", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgbcon('1', func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, r, &info)
		Chkxer("ZGBCON", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgbcon('1', func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), &ip, &anrm, &rcond, w, r, &info)
		Chkxer("ZGBCON", &info, lerr, ok, t)
		*infot = 6
		golapack.Zgbcon('1', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 3; return &y }(), &ip, &anrm, &rcond, w, r, &info)
		Chkxer("ZGBCON", &info, lerr, ok, t)

		//        ZGBEQU
		*srnamt = "ZGBEQU"
		*infot = 1
		golapack.Zgbequ(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("ZGBEQU", &info, lerr, ok, t)
		*infot = 2
		golapack.Zgbequ(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("ZGBEQU", &info, lerr, ok, t)
		*infot = 3
		golapack.Zgbequ(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("ZGBEQU", &info, lerr, ok, t)
		*infot = 4
		golapack.Zgbequ(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("ZGBEQU", &info, lerr, ok, t)
		*infot = 6
		golapack.Zgbequ(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), r1, r2, &rcond, &ccond, &anrm, &info)
		Chkxer("ZGBEQU", &info, lerr, ok, t)
	}

	//     Print a summary line.
	Alaesm(path, ok)
}
