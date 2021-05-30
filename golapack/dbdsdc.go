package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dbdsdc computes the singular value decomposition (SVD) of a real
// N-by-N (upper or lower) bidiagonal matrix B:  B = U * S * VT,
// using a divide and conquer method, where S is a diagonal matrix
// with non-negative diagonal elements (the singular values of B), and
// U and VT are orthogonal matrices of left and right singular vectors,
// respectively. DBDSDC can be used to compute all singular values,
// and optionally, singular vectors or singular vectors in compact form.
//
// This code makes very mild assumptions about floating point
// arithmetic. It will work on machines with a guard digit in
// add/subtract, or on those binary machines without guard digits
// which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
// It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.  See DLASD3 for details.
//
// The code currently calls DLASDQ if singular values only are desired.
// However, it can be slightly modified to compute singular values
// using the divide and conquer method.
func Dbdsdc(uplo, compq byte, n *int, d, e *mat.Vector, u *mat.Matrix, ldu *int, vt *mat.Matrix, ldvt *int, q *mat.Vector, iq *[]int, work *mat.Vector, iwork *[]int, info *int) {
	var cs, eps, one, orgnrm, p, r, sn, two, zero float64
	var difl, difr, givcol, givnum, givptr, i, ic, icompq, ierr, ii, is, iu, iuplo, ivt, j, k, kk, mlvl, nm1, nsize, perm, poles, qstart, smlsiz, smlszp, sqre, start, wstart, z int

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Test the input parameters.
	(*info) = 0

	iuplo = 0
	if uplo == 'U' {
		iuplo = 1
	}
	if uplo == 'L' {
		iuplo = 2
	}
	if compq == 'N' {
		icompq = 0
	} else if compq == 'P' {
		icompq = 1
	} else if compq == 'I' {
		icompq = 2
	} else {
		icompq = -1
	}
	if iuplo == 0 {
		(*info) = -1
	} else if icompq < 0 {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if ((*ldu) < 1) || ((icompq == 2) && ((*ldu) < (*n))) {
		(*info) = -7
	} else if ((*ldvt) < 1) || ((icompq == 2) && ((*ldvt) < (*n))) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DBDSDC"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}
	smlsiz = Ilaenv(func() *int { y := 9; return &y }(), []byte("DBDSDC"), []byte{' '}, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }())
	if (*n) == 1 {
		if icompq == 1 {
			q.Set(0, math.Copysign(one, d.Get(0)))
			q.Set(1+smlsiz*(*n)-1, one)
		} else if icompq == 2 {
			u.Set(0, 0, math.Copysign(one, d.Get(0)))
			vt.Set(0, 0, one)
		}
		d.Set(0, math.Abs(d.Get(0)))
		return
	}
	nm1 = (*n) - 1

	//     If matrix lower bidiagonal, rotate to be upper bidiagonal
	//     by applying Givens rotations on the left
	wstart = 1
	qstart = 3
	if icompq == 1 {
		goblas.Dcopy(n, d, func() *int { y := 1; return &y }(), q, func() *int { y := 1; return &y }())
		goblas.Dcopy(toPtr((*n)-1), e, func() *int { y := 1; return &y }(), q.Off((*n)+1-1), func() *int { y := 1; return &y }())
	}
	if iuplo == 2 {
		qstart = 5
		if icompq == 2 {
			wstart = 2*(*n) - 1
		}
		for i = 1; i <= (*n)-1; i++ {
			Dlartg(d.GetPtr(i-1), e.GetPtr(i-1), &cs, &sn, &r)
			d.Set(i-1, r)
			e.Set(i-1, sn*d.Get(i+1-1))
			d.Set(i+1-1, cs*d.Get(i+1-1))
			if icompq == 1 {
				q.Set(i+2*(*n)-1, cs)
				q.Set(i+3*(*n)-1, sn)
			} else if icompq == 2 {
				work.Set(i-1, cs)
				work.Set(nm1+i-1, -sn)
			}
		}
	}

	//     If ICOMPQ = 0, use DLASDQ to compute the singular values.
	if icompq == 0 {
		//        Ignore WSTART, instead using WORK( 1 ), since the two vectors
		//        for CS and -SN above are added only if ICOMPQ == 2,
		//        and adding them exceeds documented WORK size of 4*n.
		Dlasdq('U', func() *int { y := 0; return &y }(), n, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d, e, vt, ldvt, u, ldu, u, ldu, work, info)
		goto label40
	}

	//     If N is smaller than the minimum divide size SMLSIZ, then solve
	//     the problem with another solver.
	if (*n) <= smlsiz {
		if icompq == 2 {
			Dlaset('A', n, n, &zero, &one, u, ldu)
			Dlaset('A', n, n, &zero, &one, vt, ldvt)
			Dlasdq('U', func() *int { y := 0; return &y }(), n, n, n, func() *int { y := 0; return &y }(), d, e, vt, ldvt, u, ldu, u, ldu, work.Off(wstart-1), info)
		} else if icompq == 1 {
			iu = 1
			ivt = iu + (*n)
			Dlaset('A', n, n, &zero, &one, q.MatrixOff(iu+(qstart-1)*(*n)-1, *n, opts), n)
			Dlaset('A', n, n, &zero, &one, q.MatrixOff(ivt+(qstart-1)*(*n)-1, *n, opts), n)
			Dlasdq('U', func() *int { y := 0; return &y }(), n, n, n, func() *int { y := 0; return &y }(), d, e, q.MatrixOff(ivt+(qstart-1)*(*n)-1, *n, opts), n, q.MatrixOff(iu+(qstart-1)*(*n)-1, *n, opts), n, q.MatrixOff(iu+(qstart-1)*(*n)-1, *n, opts), n, work.Off(wstart-1), info)
		}
		goto label40
	}

	if icompq == 2 {
		Dlaset('A', n, n, &zero, &one, u, ldu)
		Dlaset('A', n, n, &zero, &one, vt, ldvt)
	}

	//     Scale.
	orgnrm = Dlanst('M', n, d, e)
	if orgnrm == zero {
		return
	}
	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, n, func() *int { y := 1; return &y }(), d.Matrix(*n, opts), n, &ierr)
	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, &nm1, func() *int { y := 1; return &y }(), e.Matrix(nm1, opts), &nm1, &ierr)

	eps = 0.9 * Dlamch(Epsilon)

	mlvl = int(math.Log(float64(*n)/float64(smlsiz+1))/math.Log(two)) + 1
	smlszp = smlsiz + 1

	if icompq == 1 {
		iu = 1
		ivt = 1 + smlsiz
		difl = ivt + smlszp
		difr = difl + mlvl
		z = difr + mlvl*2
		ic = z + mlvl
		is = ic + 1
		poles = is + 1
		givnum = poles + 2*mlvl
		//
		k = 1
		givptr = 2
		perm = 3
		givcol = perm + mlvl
	}

	for i = 1; i <= (*n); i++ {
		if math.Abs(d.Get(i-1)) < eps {
			d.Set(i-1, math.Copysign(eps, d.Get(i-1)))
		}
	}

	start = 1
	sqre = 0

	for i = 1; i <= nm1; i++ {
		if (math.Abs(e.Get(i-1)) < eps) || (i == nm1) {
			//           Subproblem found. First determine its size and then
			//           apply divide and conquer on it.
			if i < nm1 {
				//              A subproblem with E(I) small for I < NM1.
				nsize = i - start + 1
			} else if math.Abs(e.Get(i-1)) >= eps {
				//              A subproblem with E(NM1) not too small but I = NM1.
				nsize = (*n) - start + 1
			} else {
				//              A subproblem with E(NM1) small. This implies an
				//              1-by-1 subproblem at D(N). Solve this 1-by-1 problem
				//              first.
				nsize = i - start + 1
				if icompq == 2 {
					u.Set((*n)-1, (*n)-1, math.Copysign(one, d.Get((*n)-1)))
					vt.Set((*n)-1, (*n)-1, one)
				} else if icompq == 1 {
					q.Set((*n)+(qstart-1)*(*n)-1, math.Copysign(one, d.Get((*n)-1)))
					q.Set((*n)+(smlsiz+qstart-1)*(*n)-1, one)
				}
				d.Set((*n)-1, math.Abs(d.Get((*n)-1)))
			}
			if icompq == 2 {
				Dlasd0(&nsize, &sqre, d.Off(start-1), e.Off(start-1), u.Off(start-1, start-1), ldu, vt.Off(start-1, start-1), ldvt, &smlsiz, iwork, work.Off(wstart-1), info)
			} else {
				Dlasda(&icompq, &smlsiz, &nsize, &sqre, d.Off(start-1), e.Off(start-1), q.MatrixOff(start+(iu+qstart-2)*(*n)-1, *n, opts), n, q.MatrixOff(start+(ivt+qstart-2)*(*n)-1, *n, opts), toSlice(iq, start+k*(*n)-1), q.MatrixOff(start+(difl+qstart-2)*(*n)-1, *n, opts), q.MatrixOff(start+(difr+qstart-2)*(*n)-1, *n, opts), q.MatrixOff(start+(z+qstart-2)*(*n)-1, *n, opts), q.MatrixOff(start+(poles+qstart-2)*(*n)-1, *n, opts), toSlice(iq, start+givptr*(*n)-1), toSlice(iq, start+givcol*(*n)-1), n, toSlice(iq, start+perm*(*n)-1), q.MatrixOff(start+(givnum+qstart-2)*(*n)-1, *n, opts), q.Off(start+(ic+qstart-2)*(*n)-1), q.Off(start+(is+qstart-2)*(*n)-1), work.Off(wstart-1), iwork, info)
			}
			if (*info) != 0 {
				return
			}
			start = i + 1
		}
	}

	//     Unscale
	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &orgnrm, n, func() *int { y := 1; return &y }(), d.Matrix(*n, opts), n, &ierr)
label40:
	;

	//     Use Selection Sort to minimize swaps of singular vectors
	for ii = 2; ii <= (*n); ii++ {
		i = ii - 1
		kk = i
		p = d.Get(i - 1)
		for j = ii; j <= (*n); j++ {
			if d.Get(j-1) > p {
				kk = j
				p = d.Get(j - 1)
			}
		}
		if kk != i {
			d.Set(kk-1, d.Get(i-1))
			d.Set(i-1, p)
			if icompq == 1 {
				(*iq)[i-1] = kk
			} else if icompq == 2 {
				goblas.Dswap(n, u.Vector(0, i-1), func() *int { y := 1; return &y }(), u.Vector(0, kk-1), func() *int { y := 1; return &y }())
				goblas.Dswap(n, vt.Vector(i-1, 0), ldvt, vt.Vector(kk-1, 0), ldvt)
			}
		} else if icompq == 1 {
			(*iq)[i-1] = i
		}
	}

	//     If ICOMPQ = 1, use IQ(N,1) as the indicator for UPLO
	if icompq == 1 {
		if iuplo == 1 {
			(*iq)[(*n)-1] = 1
		} else {
			(*iq)[(*n)-1] = 0
		}
	}

	//     If B is lower bidiagonal, update U by those Givens rotations
	//     which rotated B to be upper bidiagonal
	if (iuplo == 2) && (icompq == 2) {
		Dlasr('L', 'V', 'B', n, n, work, work.Off((*n)-1), u, ldu)
	}
}
