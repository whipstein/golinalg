package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zlalsd uses the singular value decomposition of A to solve the least
// squares problem of finding X to minimize the Euclidean norm of each
// column of A*X-B, where A is N-by-N upper bidiagonal, and X and B
// are N-by-NRHS. The solution X overwrites B.
//
// The singular values of A smaller than RCOND times the largest
// singular value are treated as zero in solving the least squares
// problem; in this case a minimum norm solution is returned.
// The actual singular values are returned in D in ascending order.
//
// This code makes very mild assumptions about floating point
// arithmetic. It will work on machines with a guard digit in
// add/subtract, or on those binary machines without guard digits
// which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2.
// It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zlalsd(uplo byte, smlsiz, n, nrhs *int, d, e *mat.Vector, b *mat.CMatrix, ldb *int, rcond *float64, rank *int, work *mat.CVector, rwork *mat.Vector, iwork *[]int, info *int) {
	var czero complex128
	var cs, eps, one, orgnrm, r, rcnd, sn, tol, two, zero float64
	var bx, bxst, c, difl, difr, givcol, givnum, givptr, i, icmpq1, icmpq2, irwb, irwib, irwrb, irwu, irwvt, irwwrk, iwk, j, jcol, jimag, jreal, jrow, k, nlvl, nm1, nrwork, nsize, nsub, perm, poles, s, sizei, smlszp, sqre, st, st1, u, vt, z int

	zero = 0.0
	one = 1.0
	two = 2.0
	czero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0

	if (*n) < 0 {
		(*info) = -3
	} else if (*nrhs) < 1 {
		(*info) = -4
	} else if ((*ldb) < 1) || ((*ldb) < (*n)) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLALSD"), -(*info))
		return
	}

	eps = Dlamch(Epsilon)

	//     Set up the tolerance.
	if ((*rcond) <= zero) || ((*rcond) >= one) {
		rcnd = eps
	} else {
		rcnd = (*rcond)
	}

	(*rank) = 0

	//     Quick return if possible.
	if (*n) == 0 {
		return
	} else if (*n) == 1 {
		if d.Get(0) == zero {
			Zlaset('A', func() *int { y := 1; return &y }(), nrhs, &czero, &czero, b, ldb)
		} else {
			(*rank) = 1
			Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d.GetPtr(0), &one, func() *int { y := 1; return &y }(), nrhs, b, ldb, info)
			d.Set(0, d.GetMag(0))
		}
		return
	}

	//     Rotate the matrix if it is lower bidiagonal.
	if uplo == 'L' {
		for i = 1; i <= (*n)-1; i++ {
			Dlartg(d.GetPtr(i-1), e.GetPtr(i-1), &cs, &sn, &r)
			d.Set(i-1, r)
			e.Set(i-1, sn*d.Get(i+1-1))
			d.Set(i+1-1, cs*d.Get(i+1-1))
			if (*nrhs) == 1 {
				goblas.Zdrot(func() *int { y := 1; return &y }(), b.CVector(i-1, 0), func() *int { y := 1; return &y }(), b.CVector(i+1-1, 0), func() *int { y := 1; return &y }(), &cs, &sn)
			} else {
				rwork.Set(i*2-1-1, cs)
				rwork.Set(i*2-1, sn)
			}
		}
		if (*nrhs) > 1 {
			for i = 1; i <= (*nrhs); i++ {
				for j = 1; j <= (*n)-1; j++ {
					cs = rwork.Get(j*2 - 1 - 1)
					sn = rwork.Get(j*2 - 1)
					goblas.Zdrot(func() *int { y := 1; return &y }(), b.CVector(j-1, i-1), func() *int { y := 1; return &y }(), b.CVector(j+1-1, i-1), func() *int { y := 1; return &y }(), &cs, &sn)
				}
			}
		}
	}

	//     Scale.
	nm1 = (*n) - 1
	orgnrm = Dlanst('M', n, d, e)
	if orgnrm == zero {
		Zlaset('A', n, nrhs, &czero, &czero, b, ldb)
		return
	}

	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, n, func() *int { y := 1; return &y }(), d.Matrix(*n, opts), n, info)
	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, &nm1, func() *int { y := 1; return &y }(), e.Matrix(nm1, opts), &nm1, info)

	//     If N is smaller than the minimum divide size SMLSIZ, then solve
	//     the problem with another solver.
	if (*n) <= (*smlsiz) {
		irwu = 1
		irwvt = irwu + (*n)*(*n)
		irwwrk = irwvt + (*n)*(*n)
		irwrb = irwwrk
		irwib = irwrb + (*n)*(*nrhs)
		irwb = irwib + (*n)*(*nrhs)
		Dlaset('A', n, n, &zero, &one, rwork.MatrixOff(irwu-1, *n, opts), n)
		Dlaset('A', n, n, &zero, &one, rwork.MatrixOff(irwvt-1, *n, opts), n)
		Dlasdq('U', func() *int { y := 0; return &y }(), n, n, n, func() *int { y := 0; return &y }(), d, e, rwork.MatrixOff(irwvt-1, *n, opts), n, rwork.MatrixOff(irwu-1, *n, opts), n, rwork.MatrixOff(irwwrk-1, 1, opts), func() *int { y := 1; return &y }(), rwork.Off(irwwrk-1), info)
		if (*info) != 0 {
			return
		}

		//        In the real version, B is passed to DLASDQ and multiplied
		//        internally by Q**H. Here B is complex and that product is
		//        computed below in two steps (real and imaginary parts).
		j = irwb - 1
		for jcol = 1; jcol <= (*nrhs); jcol++ {
			for jrow = 1; jrow <= (*n); jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetRe(jrow-1, jcol-1))
			}
		}
		goblas.Dgemm(Trans, NoTrans, n, nrhs, n, &one, rwork.MatrixOff(irwu-1, *n, opts), n, rwork.MatrixOff(irwb-1, *n, opts), n, &zero, rwork.MatrixOff(irwrb-1, *n, opts), n)
		j = irwb - 1
		for jcol = 1; jcol <= (*nrhs); jcol++ {
			for jrow = 1; jrow <= (*n); jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetIm(jrow-1, jcol-1))
			}
		}
		goblas.Dgemm(Trans, NoTrans, n, nrhs, n, &one, rwork.MatrixOff(irwu-1, *n, opts), n, rwork.MatrixOff(irwb-1, *n, opts), n, &zero, rwork.MatrixOff(irwib-1, *n, opts), n)
		jreal = irwrb - 1
		jimag = irwib - 1
		for jcol = 1; jcol <= (*nrhs); jcol++ {
			for jrow = 1; jrow <= (*n); jrow++ {
				jreal = jreal + 1
				jimag = jimag + 1
				b.Set(jrow-1, jcol-1, complex(rwork.Get(jreal-1), rwork.Get(jimag-1)))
			}
		}

		tol = rcnd * d.GetMag(goblas.Idamax(n, d, func() *int { y := 1; return &y }())-1)
		for i = 1; i <= (*n); i++ {
			if d.Get(i-1) <= tol {
				Zlaset('A', func() *int { y := 1; return &y }(), nrhs, &czero, &czero, b.Off(i-1, 0), ldb)
			} else {
				Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d.GetPtr(i-1), &one, func() *int { y := 1; return &y }(), nrhs, b.Off(i-1, 0), ldb, info)
				(*rank) = (*rank) + 1
			}
		}

		//        Since B is complex, the following call to DGEMM is performed
		//        in two steps (real and imaginary parts). That is for V * B
		//        (in the real version of the code V**H is stored in WORK).
		//
		//        CALL DGEMM( 'T', 'N', N, NRHS, N, ONE, WORK, N, B, LDB, ZERO,
		//    $               WORK( NWORK ), N )
		j = irwb - 1
		for jcol = 1; jcol <= (*nrhs); jcol++ {
			for jrow = 1; jrow <= (*n); jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetRe(jrow-1, jcol-1))
			}
		}
		goblas.Dgemm(Trans, NoTrans, n, nrhs, n, &one, rwork.MatrixOff(irwvt-1, *n, opts), n, rwork.MatrixOff(irwb-1, *n, opts), n, &zero, rwork.MatrixOff(irwrb-1, *n, opts), n)
		j = irwb - 1
		for jcol = 1; jcol <= (*nrhs); jcol++ {
			for jrow = 1; jrow <= (*n); jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetIm(jrow-1, jcol-1))
			}
		}
		goblas.Dgemm(Trans, NoTrans, n, nrhs, n, &one, rwork.MatrixOff(irwvt-1, *n, opts), n, rwork.MatrixOff(irwb-1, *n, opts), n, &zero, rwork.MatrixOff(irwib-1, *n, opts), n)
		jreal = irwrb - 1
		jimag = irwib - 1
		for jcol = 1; jcol <= (*nrhs); jcol++ {
			for jrow = 1; jrow <= (*n); jrow++ {
				jreal = jreal + 1
				jimag = jimag + 1
				b.Set(jrow-1, jcol-1, complex(rwork.Get(jreal-1), rwork.Get(jimag-1)))
			}
		}

		//        Unscale.
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &orgnrm, n, func() *int { y := 1; return &y }(), d.Matrix(*n, opts), n, info)
		Dlasrt('D', n, d, info)
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, n, nrhs, b, ldb, info)

		return
	}

	//     Book-keeping and setting up some constants.
	nlvl = int(math.Log(float64(*n)/float64((*smlsiz)+1))/math.Log(two)) + 1

	smlszp = (*smlsiz) + 1

	u = 1
	vt = 1 + (*smlsiz)*(*n)
	difl = vt + smlszp*(*n)
	difr = difl + nlvl*(*n)
	z = difr + nlvl*(*n)*2
	c = z + nlvl*(*n)
	s = c + (*n)
	poles = s + (*n)
	givnum = poles + 2*nlvl*(*n)
	nrwork = givnum + 2*nlvl*(*n)
	bx = 1

	irwrb = nrwork
	irwib = irwrb + (*smlsiz)*(*nrhs)
	irwb = irwib + (*smlsiz)*(*nrhs)

	sizei = 1 + (*n)
	k = sizei + (*n)
	givptr = k + (*n)
	perm = givptr + (*n)
	givcol = perm + nlvl*(*n)
	iwk = givcol + nlvl*(*n)*2

	st = 1
	sqre = 0
	icmpq1 = 1
	icmpq2 = 0
	nsub = 0

	for i = 1; i <= (*n); i++ {
		if d.GetMag(i-1) < eps {
			d.Set(i-1, math.Copysign(eps, d.Get(i-1)))
		}
	}

	for i = 1; i <= nm1; i++ {
		if (e.GetMag(i-1) < eps) || (i == nm1) {
			nsub = nsub + 1
			(*iwork)[nsub-1] = st

			//           Subproblem found. First determine its size and then
			//           apply divide and conquer on it.
			if i < nm1 {
				//              A subproblem with E(I) small for I < NM1.
				nsize = i - st + 1
				(*iwork)[sizei+nsub-1-1] = nsize
			} else if e.GetMag(i-1) >= eps {
				//              A subproblem with E(NM1) not too small but I = NM1.
				nsize = (*n) - st + 1
				(*iwork)[sizei+nsub-1-1] = nsize
			} else {
				//              A subproblem with E(NM1) small. This implies an
				//              1-by-1 subproblem at D(N), which is not solved
				//              explicitly.
				nsize = i - st + 1
				(*iwork)[sizei+nsub-1-1] = nsize
				nsub = nsub + 1
				(*iwork)[nsub-1] = (*n)
				(*iwork)[sizei+nsub-1-1] = 1
				goblas.Zcopy(nrhs, b.CVector((*n)-1, 0), ldb, work.Off(bx+nm1-1), n)
			}
			st1 = st - 1
			if nsize == 1 {
				//              This is a 1-by-1 subproblem and is not solved
				//              explicitly.
				goblas.Zcopy(nrhs, b.CVector(st-1, 0), ldb, work.Off(bx+st1-1), n)
			} else if nsize <= (*smlsiz) {
				//              This is a small subproblem and is solved by DLASDQ.
				Dlaset('A', &nsize, &nsize, &zero, &one, rwork.MatrixOff(vt+st1-1, *n, opts), n)
				Dlaset('A', &nsize, &nsize, &zero, &one, rwork.MatrixOff(u+st1-1, *n, opts), n)
				Dlasdq('U', func() *int { y := 0; return &y }(), &nsize, &nsize, &nsize, func() *int { y := 0; return &y }(), d.Off(st-1), e.Off(st-1), rwork.MatrixOff(vt+st1-1, *n, opts), n, rwork.MatrixOff(u+st1-1, *n, opts), n, rwork.MatrixOff(nrwork-1, 1, opts), func() *int { y := 1; return &y }(), rwork.Off(nrwork-1), info)
				if (*info) != 0 {
					return
				}

				//              In the real version, B is passed to DLASDQ and multiplied
				//              internally by Q**H. Here B is complex and that product is
				//              computed below in two steps (real and imaginary parts).
				j = irwb - 1
				for jcol = 1; jcol <= (*nrhs); jcol++ {
					for jrow = st; jrow <= st+nsize-1; jrow++ {
						j = j + 1
						rwork.Set(j-1, b.GetRe(jrow-1, jcol-1))
					}
				}
				goblas.Dgemm(Trans, NoTrans, &nsize, nrhs, &nsize, &one, rwork.MatrixOff(u+st1-1, *n, opts), n, rwork.MatrixOff(irwb-1, nsize, opts), &nsize, &zero, rwork.MatrixOff(irwrb-1, nsize, opts), &nsize)
				j = irwb - 1
				for jcol = 1; jcol <= (*nrhs); jcol++ {
					for jrow = st; jrow <= st+nsize-1; jrow++ {
						j = j + 1
						rwork.Set(j-1, b.GetIm(jrow-1, jcol-1))
					}
				}
				goblas.Dgemm(Trans, NoTrans, &nsize, nrhs, &nsize, &one, rwork.MatrixOff(u+st1-1, *n, opts), n, rwork.MatrixOff(irwb-1, nsize, opts), &nsize, &zero, rwork.MatrixOff(irwib-1, nsize, opts), &nsize)
				jreal = irwrb - 1
				jimag = irwib - 1
				for jcol = 1; jcol <= (*nrhs); jcol++ {
					for jrow = st; jrow <= st+nsize-1; jrow++ {
						jreal = jreal + 1
						jimag = jimag + 1
						b.Set(jrow-1, jcol-1, complex(rwork.Get(jreal-1), rwork.Get(jimag-1)))
					}
				}

				Zlacpy('A', &nsize, nrhs, b.Off(st-1, 0), ldb, work.CMatrixOff(bx+st1-1, *n, opts), n)
			} else {
				//              A large problem. Solve it using divide and conquer.
				Dlasda(&icmpq1, smlsiz, &nsize, &sqre, d.Off(st-1), e.Off(st-1), rwork.MatrixOff(u+st1-1, *n, opts), n, rwork.MatrixOff(vt+st1-1, *n, opts), toSlice(iwork, k+st1-1), rwork.MatrixOff(difl+st1-1, *n, opts), rwork.MatrixOff(difr+st1-1, *n, opts), rwork.MatrixOff(z+st1-1, *n, opts), rwork.MatrixOff(poles+st1-1, *n, opts), toSlice(iwork, givptr+st1-1), toSlice(iwork, givcol+st1-1), n, toSlice(iwork, perm+st1-1), rwork.MatrixOff(givnum+st1-1, *n, opts), rwork.Off(c+st1-1), rwork.Off(s+st1-1), rwork.Off(nrwork-1), toSlice(iwork, iwk-1), info)
				if (*info) != 0 {
					return
				}
				bxst = bx + st1
				Zlalsa(&icmpq2, smlsiz, &nsize, nrhs, b.Off(st-1, 0), ldb, work.CMatrixOff(bxst-1, *n, opts), n, rwork.MatrixOff(u+st1-1, *n, opts), n, rwork.MatrixOff(vt+st1-1, *n, opts), toSlice(iwork, k+st1-1), rwork.MatrixOff(difl+st1-1, *n, opts), rwork.MatrixOff(difr+st1-1, *n, opts), rwork.MatrixOff(z+st1-1, *n, opts), rwork.MatrixOff(poles+st1-1, *n, opts), toSlice(iwork, givptr+st1-1), toSlice(iwork, givcol+st1-1), n, toSlice(iwork, perm+st1-1), rwork.MatrixOff(givnum+st1-1, *n, opts), rwork.Off(c+st1-1), rwork.Off(s+st1-1), rwork.Off(nrwork-1), toSlice(iwork, iwk-1), info)
				if (*info) != 0 {
					return
				}
			}
			st = i + 1
		}
	}

	//     Apply the singular values and treat the tiny ones as zero.
	tol = rcnd * d.GetMag(goblas.Idamax(n, d, func() *int { y := 1; return &y }())-1)

	for i = 1; i <= (*n); i++ {
		//        Some of the elements in D can be negative because 1-by-1
		//        subproblems were not solved explicitly.
		if d.GetMag(i-1) <= tol {
			Zlaset('A', func() *int { y := 1; return &y }(), nrhs, &czero, &czero, work.CMatrixOff(bx+i-1-1, *n, opts), n)
		} else {
			(*rank) = (*rank) + 1
			Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d.GetPtr(i-1), &one, func() *int { y := 1; return &y }(), nrhs, work.CMatrixOff(bx+i-1-1, *n, opts), n, info)
		}
		d.Set(i-1, d.GetMag(i-1))
	}

	//     Now apply back the right singular vectors.
	icmpq2 = 1
	for i = 1; i <= nsub; i++ {
		st = (*iwork)[i-1]
		st1 = st - 1
		nsize = (*iwork)[sizei+i-1-1]
		bxst = bx + st1
		if nsize == 1 {
			goblas.Zcopy(nrhs, work.Off(bxst-1), n, b.CVector(st-1, 0), ldb)
		} else if nsize <= (*smlsiz) {
			//           Since B and BX are complex, the following call to DGEMM
			//           is performed in two steps (real and imaginary parts).
			//
			//           CALL DGEMM( 'T', 'N', NSIZE, NRHS, NSIZE, ONE,
			//    $                  RWORK( VT+ST1 ), N, RWORK( BXST ), N, ZERO,
			//    $                  B( ST, 1 ), LDB )
			j = bxst - (*n) - 1
			jreal = irwb - 1
			for jcol = 1; jcol <= (*nrhs); jcol++ {
				j = j + (*n)
				for jrow = 1; jrow <= nsize; jrow++ {
					jreal = jreal + 1
					rwork.Set(jreal-1, work.GetRe(j+jrow-1))
				}
			}
			goblas.Dgemm(Trans, NoTrans, &nsize, nrhs, &nsize, &one, rwork.MatrixOff(vt+st1-1, *n, opts), n, rwork.MatrixOff(irwb-1, nsize, opts), &nsize, &zero, rwork.MatrixOff(irwrb-1, nsize, opts), &nsize)
			j = bxst - (*n) - 1
			jimag = irwb - 1
			for jcol = 1; jcol <= (*nrhs); jcol++ {
				j = j + (*n)
				for jrow = 1; jrow <= nsize; jrow++ {
					jimag = jimag + 1
					rwork.Set(jimag-1, work.GetIm(j+jrow-1))
				}
			}
			goblas.Dgemm(Trans, NoTrans, &nsize, nrhs, &nsize, &one, rwork.MatrixOff(vt+st1-1, *n, opts), n, rwork.MatrixOff(irwb-1, nsize, opts), &nsize, &zero, rwork.MatrixOff(irwib-1, nsize, opts), &nsize)
			jreal = irwrb - 1
			jimag = irwib - 1
			for jcol = 1; jcol <= (*nrhs); jcol++ {
				for jrow = st; jrow <= st+nsize-1; jrow++ {
					jreal = jreal + 1
					jimag = jimag + 1
					b.Set(jrow-1, jcol-1, complex(rwork.Get(jreal-1), rwork.Get(jimag-1)))
				}
			}
		} else {
			Zlalsa(&icmpq2, smlsiz, &nsize, nrhs, work.CMatrixOff(bxst-1, *n, opts), n, b.Off(st-1, 0), ldb, rwork.MatrixOff(u+st1-1, *n, opts), n, rwork.MatrixOff(vt+st1-1, *n, opts), toSlice(iwork, k+st1-1), rwork.MatrixOff(difl+st1-1, *n, opts), rwork.MatrixOff(difr+st1-1, *n, opts), rwork.MatrixOff(z+st1-1, *n, opts), rwork.MatrixOff(poles+st1-1, *n, opts), toSlice(iwork, givptr+st1-1), toSlice(iwork, givcol+st1-1), n, toSlice(iwork, perm+st1-1), rwork.MatrixOff(givnum+st1-1, *n, opts), rwork.Off(c+st1-1), rwork.Off(s+st1-1), rwork.Off(nrwork-1), toSlice(iwork, iwk-1), info)
			if (*info) != 0 {
				return
			}
		}
	}

	//     Unscale and sort the singular values.
	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &orgnrm, n, func() *int { y := 1; return &y }(), d.Matrix(*n, opts), n, info)
	Dlasrt('D', n, d, info)
	Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, n, nrhs, b, ldb, info)
}
