package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetri3x computes the inverse of a complex Hermitian indefinite
// matrix A using the factorization computed by ZHETRF_RK or ZHETRF_BK:
//
//     A = P*U*D*(U**H)*(P**T) or A = P*L*D*(L**H)*(P**T),
//
// where U (or L) is unit upper (or lower) triangular matrix,
// U**H (or L**H) is the conjugate of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is Hermitian and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// This is the blocked version of the algorithm, calling Level 3 BLAS.
func Zhetri3x(uplo mat.MatUplo, n int, a *mat.CMatrix, e *mat.CVector, ipiv *[]int, work *mat.CMatrix, nb int) (info int, err error) {
	var upper bool
	var akkp1, cone, czero, d, u01IJ, u01Ip1J, u11IJ, u11Ip1J complex128
	var ak, akp1, one, t float64
	var cut, i, icount, invd, ip, j, k, nnb, u11 int

	one = 1.0
	cone = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}

	//     Quick return if possible
	if err != nil {
		gltest.Xerbla2("Zhetri3x", err)
		return
	}
	if n == 0 {
		return
	}

	//     Workspace got Non-diag elements of D
	for k = 1; k <= n; k++ {
		work.Set(k-1, 0, e.Get(k-1))
	}

	//     Check that the diagonal matrix D is nonsingular.
	if upper {
		//        Upper triangular storage: examine D from bottom to top
		for info = n; info >= 1; info-- {
			if (*ipiv)[info-1] > 0 && a.Get(info-1, info-1) == czero {
				return
			}
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		for info = 1; info <= n; info++ {
			if (*ipiv)[info-1] > 0 && a.Get(info-1, info-1) == czero {
				return
			}
		}
	}

	info = 0

	//     Splitting Workspace
	//     U01 is a block ( N, NB+1 )
	//     The first element of U01 is in WORK( 1, 1 )
	//     U11 is a block ( NB+1, NB+1 )
	//     The first element of U11 is in WORK( N+1, 1 )
	u11 = n

	//     INVD is a block ( N, 2 )
	//     The first element of INVD is in WORK( 1, INVD )
	invd = nb + 2
	if upper {
		//        Begin Upper
		//
		//        invA = P * inv(U**H) * inv(D) * inv(U) * P**T.
		if info, err = Ztrtri(uplo, Unit, n, a); err != nil {
			panic(err)
		}

		//        inv(D) and inv(D) * inv(U)
		k = 1
		for k <= n {
			if (*ipiv)[k-1] > 0 {
				//              1 x 1 diagonal NNB
				work.SetRe(k-1, invd-1, one/a.GetRe(k-1, k-1))
				work.Set(k-1, invd, czero)
			} else {
				//              2 x 2 diagonal NNB
				t = work.GetMag(k, 0)
				ak = a.GetRe(k-1, k-1) / t
				akp1 = a.GetRe(k, k) / t
				akkp1 = work.Get(k, 0) / complex(t, 0)
				d = complex(t, 0) * (complex(ak*akp1, 0) - cone)
				work.Set(k-1, invd-1, complex(akp1, 0)/d)
				work.Set(k, invd, complex(ak, 0)/d)
				work.Set(k-1, invd, -akkp1/d)
				work.Set(k, invd-1, work.GetConj(k-1, invd))
				k = k + 1
			}
			k = k + 1
		}

		//        inv(U**H) = (inv(U))**H
		//
		//        inv(U**H) * inv(D) * inv(U)
		cut = n
		for cut > 0 {
			nnb = nb
			if cut <= nnb {
				nnb = cut
			} else {
				icount = 0
				//              count negative elements,
				for i = cut + 1 - nnb; i <= cut; i++ {
					if (*ipiv)[i-1] < 0 {
						icount = icount + 1
					}
				}
				//              need a even number for a clear cut
				if (icount % 2) == 1 {
					nnb = nnb + 1
				}
			}
			cut = cut - nnb

			//           U01 Block
			for i = 1; i <= cut; i++ {
				for j = 1; j <= nnb; j++ {
					work.Set(i-1, j-1, a.Get(i-1, cut+j-1))
				}
			}

			//           U11 Block
			for i = 1; i <= nnb; i++ {
				work.Set(u11+i-1, i-1, cone)
				for j = 1; j <= i-1; j++ {
					work.Set(u11+i-1, j-1, czero)
				}
				for j = i + 1; j <= nnb; j++ {
					work.Set(u11+i-1, j-1, a.Get(cut+i-1, cut+j-1))
				}
			}

			//           invD * U01
			i = 1
			for i <= cut {
				if (*ipiv)[i-1] > 0 {
					for j = 1; j <= nnb; j++ {
						work.Set(i-1, j-1, work.Get(i-1, invd-1)*work.Get(i-1, j-1))
					}
				} else {
					for j = 1; j <= nnb; j++ {
						u01IJ = work.Get(i-1, j-1)
						u01Ip1J = work.Get(i, j-1)
						work.Set(i-1, j-1, work.Get(i-1, invd-1)*u01IJ+work.Get(i-1, invd)*u01Ip1J)
						work.Set(i, j-1, work.Get(i, invd-1)*u01IJ+work.Get(i, invd)*u01Ip1J)
					}
					i = i + 1
				}
				i = i + 1
			}

			//           invD1 * U11
			i = 1
			for i <= nnb {
				if (*ipiv)[cut+i-1] > 0 {
					for j = i; j <= nnb; j++ {
						work.Set(u11+i-1, j-1, work.Get(cut+i-1, invd-1)*work.Get(u11+i-1, j-1))
					}
				} else {
					for j = i; j <= nnb; j++ {
						u11IJ = work.Get(u11+i-1, j-1)
						u11Ip1J = work.Get(u11+i, j-1)
						work.Set(u11+i-1, j-1, work.Get(cut+i-1, invd-1)*work.Get(u11+i-1, j-1)+work.Get(cut+i-1, invd)*work.Get(u11+i, j-1))
						work.Set(u11+i, j-1, work.Get(cut+i, invd-1)*u11IJ+work.Get(cut+i, invd)*u11Ip1J)
					}
					i = i + 1
				}
				i = i + 1
			}

			//           U11**H * invD1 * U11 -> U11
			if err = work.Off(u11, 0).UpdateRows(n+nb+1).Trmm(Left, Upper, ConjTrans, Unit, nnb, nnb, cone, a.Off(cut, cut)); err != nil {
				panic(err)
			}

			for i = 1; i <= nnb; i++ {
				for j = i; j <= nnb; j++ {
					a.Set(cut+i-1, cut+j-1, work.Get(u11+i-1, j-1))
				}
			}

			//           U01**H * invD * U01 -> A( CUT+I, CUT+J )
			if err = work.Off(u11, 0).UpdateRows(n+nb+1).Gemm(ConjTrans, NoTrans, nnb, nnb, cut, cone, a.Off(0, cut), work.Off(0, 0).UpdateRows(n+nb+1), czero); err != nil {
				panic(err)
			}

			//           U11 =  U11**H * invD1 * U11 + U01**H * invD * U01
			for i = 1; i <= nnb; i++ {
				for j = i; j <= nnb; j++ {
					a.Set(cut+i-1, cut+j-1, a.Get(cut+i-1, cut+j-1)+work.Get(u11+i-1, j-1))
				}
			}

			//           U01 =  U00**H * invD0 * U01
			if err = work.Off(0, 0).UpdateRows(n+nb+1).Trmm(Left, uplo, ConjTrans, Unit, cut, nnb, cone, a); err != nil {
				panic(err)
			}

			//           Update U01
			for i = 1; i <= cut; i++ {
				for j = 1; j <= nnb; j++ {
					a.Set(i-1, cut+j-1, work.Get(i-1, j-1))
				}
			}

			//           Next Block
		}

		//        Apply PERMUTATIONS P and P**T:
		//        P * inv(U**H) * inv(D) * inv(U) * P**T.
		//        Interchange rows and columns I and IPIV(I) in reverse order
		//        from the formation order of IPIV vector for Upper case.
		//
		//        ( We can use a loop over IPIV with increment 1,
		//        since the ABS value of IPIV(I) represents the row (column)
		//        index of the interchange with row (column) i in both 1x1
		//        and 2x2 pivot cases, i.e. we don't need separate code branches
		//        for 1x1 and 2x2 pivot cases )
		for i = 1; i <= n; i++ {
			ip = abs((*ipiv)[i-1])
			if ip != i {
				if i < ip {
					Zheswapr(uplo, n, a, i, ip)
				}
				if i > ip {
					Zheswapr(uplo, n, a, ip, i)
				}
			}
		}

	} else {
		//        Begin Lower
		//
		//        inv A = P * inv(L**H) * inv(D) * inv(L) * P**T.
		if info, err = Ztrtri(uplo, Unit, n, a); err != nil {
			panic(err)
		}

		//        inv(D) and inv(D) * inv(L)
		k = n
		for k >= 1 {
			if (*ipiv)[k-1] > 0 {
				//              1 x 1 diagonal NNB
				work.SetRe(k-1, invd-1, one/a.GetRe(k-1, k-1))
				work.Set(k-1, invd, czero)
			} else {
				//              2 x 2 diagonal NNB
				t = work.GetMag(k-1-1, 0)
				ak = a.GetRe(k-1-1, k-1-1) / t
				akp1 = a.GetRe(k-1, k-1) / t
				akkp1 = work.Get(k-1-1, 0) / complex(t, 0)
				d = complex(t, 0) * (complex(ak*akp1, 0) - cone)
				work.Set(k-1-1, invd-1, complex(akp1, 0)/d)
				work.Set(k-1, invd-1, complex(ak, 0)/d)
				work.Set(k-1, invd, -akkp1/d)
				work.Set(k-1-1, invd, work.GetConj(k-1, invd))
				k = k - 1
			}
			k = k - 1
		}

		//        inv(L**H) = (inv(L))**H
		//
		//        inv(L**H) * inv(D) * inv(L)
		cut = 0
		for cut < n {
			nnb = nb
			if (cut + nnb) > n {
				nnb = n - cut
			} else {
				icount = 0
				//              count negative elements,
				for i = cut + 1; i <= cut+nnb; i++ {
					if (*ipiv)[i-1] < 0 {
						icount = icount + 1
					}
				}
				//              need a even number for a clear cut
				if (icount % 2) == 1 {
					nnb = nnb + 1
				}
			}

			//           L21 Block
			for i = 1; i <= n-cut-nnb; i++ {
				for j = 1; j <= nnb; j++ {
					work.Set(i-1, j-1, a.Get(cut+nnb+i-1, cut+j-1))
				}
			}

			//           L11 Block
			for i = 1; i <= nnb; i++ {
				work.Set(u11+i-1, i-1, cone)
				for j = i + 1; j <= nnb; j++ {
					work.Set(u11+i-1, j-1, czero)
				}
				for j = 1; j <= i-1; j++ {
					work.Set(u11+i-1, j-1, a.Get(cut+i-1, cut+j-1))
				}
			}

			//           invD*L21
			i = n - cut - nnb
			for i >= 1 {
				if (*ipiv)[cut+nnb+i-1] > 0 {
					for j = 1; j <= nnb; j++ {
						work.Set(i-1, j-1, work.Get(cut+nnb+i-1, invd-1)*work.Get(i-1, j-1))
					}
				} else {
					for j = 1; j <= nnb; j++ {
						u01IJ = work.Get(i-1, j-1)
						u01Ip1J = work.Get(i-1-1, j-1)
						work.Set(i-1, j-1, work.Get(cut+nnb+i-1, invd-1)*u01IJ+work.Get(cut+nnb+i-1, invd)*u01Ip1J)
						work.Set(i-1-1, j-1, work.Get(cut+nnb+i-1-1, invd)*u01IJ+work.Get(cut+nnb+i-1-1, invd-1)*u01Ip1J)
					}
					i = i - 1
				}
				i = i - 1
			}

			//           invD1*L11
			i = nnb
			for i >= 1 {
				if (*ipiv)[cut+i-1] > 0 {
					for j = 1; j <= nnb; j++ {
						work.Set(u11+i-1, j-1, work.Get(cut+i-1, invd-1)*work.Get(u11+i-1, j-1))
					}
				} else {
					for j = 1; j <= nnb; j++ {
						u11IJ = work.Get(u11+i-1, j-1)
						u11Ip1J = work.Get(u11+i-1-1, j-1)
						work.Set(u11+i-1, j-1, work.Get(cut+i-1, invd-1)*work.Get(u11+i-1, j-1)+work.Get(cut+i-1, invd)*u11Ip1J)
						work.Set(u11+i-1-1, j-1, work.Get(cut+i-1-1, invd)*u11IJ+work.Get(cut+i-1-1, invd-1)*u11Ip1J)
					}
					i = i - 1
				}
				i = i - 1
			}

			//           L11**H * invD1 * L11 -> L11
			if err = work.Off(u11, 0).UpdateRows(n+nb+1).Trmm(Left, uplo, ConjTrans, Unit, nnb, nnb, cone, a.Off(cut, cut)); err != nil {
				panic(err)
			}

			for i = 1; i <= nnb; i++ {
				for j = 1; j <= i; j++ {
					a.Set(cut+i-1, cut+j-1, work.Get(u11+i-1, j-1))
				}
			}

			if (cut + nnb) < n {
				//              L21**H * invD2*L21 -> A( CUT+I, CUT+J )
				if err = work.Off(u11, 0).UpdateRows(n+nb+1).Gemm(ConjTrans, NoTrans, nnb, nnb, n-nnb-cut, cone, a.Off(cut+nnb, cut), work.Off(0, 0).UpdateRows(n+nb+1), czero); err != nil {
					panic(err)
				}

				//              L11 =  L11**H * invD1 * L11 + U01**H * invD * U01
				for i = 1; i <= nnb; i++ {
					for j = 1; j <= i; j++ {
						a.Set(cut+i-1, cut+j-1, a.Get(cut+i-1, cut+j-1)+work.Get(u11+i-1, j-1))
					}
				}

				//              L01 =  L22**H * invD2 * L21
				if err = work.Trmm(Left, uplo, ConjTrans, Unit, n-nnb-cut, nnb, cone, a.Off(cut+nnb, cut+nnb)); err != nil {
					panic(err)
				}

				//              Update L21
				for i = 1; i <= n-cut-nnb; i++ {
					for j = 1; j <= nnb; j++ {
						a.Set(cut+nnb+i-1, cut+j-1, work.Get(i-1, j-1))
					}
				}

			} else {
				//              L11 =  L11**H * invD1 * L11
				for i = 1; i <= nnb; i++ {
					for j = 1; j <= i; j++ {
						a.Set(cut+i-1, cut+j-1, work.Get(u11+i-1, j-1))
					}
				}
			}

			//           Next Block
			cut = cut + nnb

		}

		//        Apply PERMUTATIONS P and P**T:
		//        P * inv(L**H) * inv(D) * inv(L) * P**T.
		//        Interchange rows and columns I and IPIV(I) in reverse order
		//        from the formation order of IPIV vector for Lower case.
		//
		//        ( We can use a loop over IPIV with increment -1,
		//        since the ABS value of IPIV(I) represents the row (column)
		//        index of the interchange with row (column) i in both 1x1
		//        and 2x2 pivot cases, i.e. we don't need separate code branches
		//        for 1x1 and 2x2 pivot cases )
		for i = n; i >= 1; i-- {
			ip = abs((*ipiv)[i-1])
			if ip != i {
				if i < ip {
					Zheswapr(uplo, n, a, i, ip)
				}
				if i > ip {
					Zheswapr(uplo, n, a, ip, i)
				}
			}
		}

	}

	return
}
