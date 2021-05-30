package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zlarhs chooses a set of NRHS random solution vectors and sets
// up the right hand sides for the linear system
//    op( A ) * X = B,
// where op( A ) may be A, A**T (transpose of A), or A**H (conjugate
// transpose of A).
func Zlarhs(path []byte, xtype, uplo, trans byte, m, n, kl, ku, nrhs *int, a *mat.CMatrix, lda *int, x *mat.CMatrix, ldx *int, b *mat.CMatrix, ldb *int, iseed *[]int, info *int) {
	var band, gen, notran, qrs, sym, tran, tri bool
	var diag byte
	var one, zero complex128
	var j, mb, nx int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	c1 := path[0]
	c2 := path[1:3]
	tran = trans == 'T' || trans == 'C'
	notran = !tran
	gen = path[1] == 'G'
	qrs = path[1] == 'Q' || path[2] == 'Q'
	sym = path[1] == 'P' || path[1] == 'S' || path[1] == 'H'
	tri = path[1] == 'T'
	band = path[2] == 'B'
	if c1 != 'Z' {
		(*info) = -1
	} else if !(xtype == 'N' || xtype == 'C') {
		(*info) = -2
	} else if (sym || tri) && !(uplo == 'U' || uplo == 'L') {
		(*info) = -3
	} else if (gen || qrs) && !(tran || trans == 'N') {
		(*info) = -4
	} else if (*m) < 0 {
		(*info) = -5
	} else if (*n) < 0 {
		(*info) = -6
	} else if band && (*kl) < 0 {
		(*info) = -7
	} else if band && (*ku) < 0 {
		(*info) = -8
	} else if (*nrhs) < 0 {
		(*info) = -9
	} else if (!band && (*lda) < maxint(1, *m)) || (band && (sym || tri) && (*lda) < (*kl)+1) || (band && gen && (*lda) < (*kl)+(*ku)+1) {
		(*info) = -11
	} else if (notran && (*ldx) < maxint(1, *n)) || (tran && (*ldx) < maxint(1, *m)) {
		(*info) = -13
	} else if (notran && (*ldb) < maxint(1, *m)) || (tran && (*ldb) < maxint(1, *n)) {
		(*info) = -15
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLARHS"), -(*info))
		return
	}

	//     Initialize X to NRHS random vectors unless XTYPE = 'C'.
	if tran {
		nx = (*m)
		mb = (*n)
	} else {
		nx = (*n)
		mb = (*m)
	}
	if xtype != 'C' {
		for j = 1; j <= (*nrhs); j++ {
			golapack.Zlarnv(func() *int { y := 2; return &y }(), iseed, n, x.CVector(0, j-1))
		}
	}

	//     Multiply X by op( A ) using an appropriate
	//     matrix multiply routine.
	if string(c2) == "GE" || string(c2) == "QR" || string(c2) == "LQ" || string(c2) == "QL" || string(c2) == "RQ" {
		//        General matrix
		goblas.Zgemm(mat.TransByte(trans), NoTrans, &mb, nrhs, &nx, &one, a, lda, x, ldx, &zero, b, ldb)

	} else if string(c2) == "PO" || string(c2) == "HE" {
		//        Hermitian matrix, 2-D storage
		goblas.Zhemm(Left, mat.UploByte(uplo), n, nrhs, &one, a, lda, x, ldx, &zero, b, ldb)

	} else if string(c2) == "SY" {
		//        Symmetric matrix, 2-D storage
		goblas.Zsymm(Left, mat.UploByte(uplo), n, nrhs, &one, a, lda, x, ldx, &zero, b, ldb)

	} else if string(c2) == "GB" {
		//        General matrix, band storage
		for j = 1; j <= (*nrhs); j++ {
			goblas.Zgbmv(mat.TransByte(trans), m, n, kl, ku, &one, a, lda, x.CVector(0, j-1), func() *int { y := 1; return &y }(), &zero, b.CVector(0, j-1), func() *int { y := 1; return &y }())
		}

	} else if string(c2) == "PB" || string(c2) == "HB" {
		//        Hermitian matrix, band storage
		for j = 1; j <= (*nrhs); j++ {
			goblas.Zhbmv(mat.UploByte(uplo), n, kl, &one, a, lda, x.CVector(0, j-1), func() *int { y := 1; return &y }(), &zero, b.CVector(0, j-1), func() *int { y := 1; return &y }())
		}

	} else if string(c2) == "SB" {
		//        Symmetric matrix, band storage
		for j = 1; j <= (*nrhs); j++ {
			Zsbmv(uplo, n, kl, &one, a, lda, x.CVector(0, j-1), func() *int { y := 1; return &y }(), &zero, b.CVector(0, j-1), func() *int { y := 1; return &y }())
		}

	} else if string(c2) == "PP" || string(c2) == "HP" {
		//        Hermitian matrix, packed storage
		for j = 1; j <= (*nrhs); j++ {
			goblas.Zhpmv(mat.UploByte(uplo), n, &one, a.CVectorIdx(0), x.CVector(0, j-1), func() *int { y := 1; return &y }(), &zero, b.CVector(0, j-1), func() *int { y := 1; return &y }())
		}

	} else if string(c2) == "SP" {
		//        Symmetric matrix, packed storage
		for j = 1; j <= (*nrhs); j++ {
			golapack.Zspmv(uplo, n, &one, a.CVectorIdx(0), x.CVector(0, j-1), func() *int { y := 1; return &y }(), &zero, b.CVector(0, j-1), func() *int { y := 1; return &y }())
		}

	} else if string(c2) == "TR" {
		//        Triangular matrix.  Note that for triangular matrices,
		//           KU = 1 => non-unit triangular
		//           KU = 2 => unit triangular
		golapack.Zlacpy('F', n, nrhs, x, ldx, b, ldb)
		if (*ku) == 2 {
			diag = 'U'
		} else {
			diag = 'N'
		}
		goblas.Ztrmm(Left, mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), n, nrhs, &one, a, lda, b, ldb)

	} else if string(c2) == "TP" {
		//        Triangular matrix, packed storage
		golapack.Zlacpy('F', n, nrhs, x, ldx, b, ldb)
		if (*ku) == 2 {
			diag = 'U'
		} else {
			diag = 'N'
		}
		for j = 1; j <= (*nrhs); j++ {
			goblas.Ztpmv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), n, a.CVectorIdx(0), b.CVector(0, j-1), func() *int { y := 1; return &y }())
		}

	} else if string(c2) == "TB" {
		//        Triangular matrix, banded storage
		golapack.Zlacpy('F', n, nrhs, x, ldx, b, ldb)
		if (*ku) == 2 {
			diag = 'U'
		} else {
			diag = 'N'
		}
		for j = 1; j <= (*nrhs); j++ {
			goblas.Ztbmv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), n, kl, a, lda, b.CVector(0, j-1), func() *int { y := 1; return &y }())
		}

	} else {
		//        If none of the above, set INFO = -1 and return
		(*info) = -1
		gltest.Xerbla([]byte("ZLARHS"), -(*info))
	}
}
