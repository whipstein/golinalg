package goblas

import (
	"fmt"
	"golinalg/mat"
	"math"
	"reflect"
	"testing"
)

func TestDblasLevel2(t *testing.T) {
	var fatal, null, reset, same, tran, upper bool
	var trans, transs mat.MatTrans
	var diag, diags mat.MatDiag
	var uplo, uplos mat.MatUplo
	var alpha, als, beta, bls, err, errmax float64
	var i, j, n int
	var a, aa, as *mat.Matrix
	var g, x, xs, xt, xx, y, ys, yt, yy *mat.Vector
	ok := true
	idim := []int{0, 1, 2, 3, 5, 9}
	kb := []int{0, 1, 2, 4}
	inc := []int{1, 2, -1, -2}
	alf := []float64{0.0, 1.0, 0.7}
	bet := []float64{0.0, 1.0, 0.9}
	thresh := 16.0
	eps := epsilonf64()
	nmax := 65
	nkb := 4
	snames := []string{"Dgemv", "Dgbmv", "Dsymv", "Dsbmv", "Dspmv", "Dtrmv", "Dtbmv", "Dtpmv", "Dtrsv", "Dtbsv", "Dtpsv", "Dger", "Dsyr", "Dspr", "Dsyr2", "Dspr2"}
	isame := make([]bool, 13)
	ichd := []mat.MatDiag{mat.Unit, mat.NonUnit}
	icht := []mat.MatTrans{mat.NoTrans, mat.Trans, mat.ConjTrans}
	ichu := []mat.MatUplo{mat.Upper, mat.Lower}
	optsfull := opts.DeepCopy()
	optsge := opts.DeepCopy()
	a = mf(nmax, nmax, opts)
	g = vf(nmax)
	x = vf(nmax)
	xt = vf(nmax)
	y = vf(nmax)
	yt = vf(nmax)
	yy = vf(nmax)

	n = minint(32, nmax)
	for j = 1; j <= n; j++ {
		for i = 1; i <= n; i++ {
			a.Set(i-1, j-1, float64(maxint(i-j+1, 0)))
		}
		x.Set(j-1, float64(j))
		y.Set(j-1, 0)
	}
	for j = 1; j <= n; j++ {
		yy.Set(j-1, float64(j*((j+1)*j))/2-float64(((j+1)*j*(j-1)))/3)
	}
	//     YY holds the exact result. On exit from SMVCH YT holds
	//     the result computed by SMVCH.
	dmvch(mat.NoTrans, n, n, 1.0, a, nmax, x, 1, 0.0, y, 1, yt, g, yy, eps, &err, &fatal, true, t)
	if err != 0 {
		t.Errorf(" ERROR IN DMVCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n DMVCH WAS CALLED WITH TRANS = %c AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", trans, same, err)
	}
	dmvch(mat.Trans, n, n, 1.0, a, nmax, x, -1, 0.0, y, -1, yt, g, yy, eps, &err, &fatal, true, t)
	if err != 0 {
		t.Errorf(" ERROR IN DMVCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n DMVCH WAS CALLED WITH TRANS = %c AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", trans, same, err)
	}
	for _, sname := range snames {
		fatal = false
		reset = true
		ok = true
		errmax = 0.0
		if sname == "Dgemv" || sname == "Dgbmv" {
			var i, iku, im, incx, incxs, incy, incys, kl, kls, ku, kus, laa, lda, ldas, m, ml, ms, n, nargs, nc, nd, nk, nl, ns int
			_ = laa
			var full bool = sname[2] == 'e'
			var banded bool = sname[2] == 'b'

			for i := range isame {
				isame[i] = true
			}

			opts.Style = mat.General
			optsfull.Style = mat.General
			if full {
				nargs = 11
				opts.Storage = mat.Dense
			} else if banded {
				nargs = 13
				opts.Storage = mat.Banded
			}

			nc = 0

			for _, n = range idim {
				nd = n/2 + 1

				for im = 1; im <= 2; im++ {
					if im == 1 {
						m = maxint(n-nd, 0)
					}
					if im == 2 {
						m = minint(n+nd, nmax)
					}

					if banded {
						nk = nkb
					} else {
						nk = 1
					}

					for iku = 1; iku <= nk; iku++ {
						if banded {
							ku = kb[iku-1]
							kl = maxint(ku-1, 0)
						} else {
							ku = n - 1
							kl = m - 1
						}
						opts.Kl, opts.Ku = kl, ku

						//              Set LDA to 1 more than minimum value if room.
						if banded {
							lda = kl + ku + 1
						} else {
							lda = m
						}
						if lda < nmax {
							lda++
						}
						//              Skip tests if not enough room.
						if lda > nmax {
							continue
						}
						laa = lda * n
						null = n <= 0 || m <= 0

						//              Generate the matrix A.
						a = mf(nmax, n, optsfull)
						aa = mf(lda, n, opts)
						dmakeL2M(m, n, a, nmax, aa, lda, kl, ku, &reset, 0.0)

						for _, trans := range mat.TransIter() {
							tran = trans.IsTrans()

							if tran {
								ml = n
								nl = m
							} else {
								ml = m
								nl = n
							}

							for _, incx = range inc {
								lx := absint(incx) * nl

								//                    Generate the vector X.
								x = vf(nl)
								xx = vf(lx)
								dmakeL2V(1, nl, x, 1, xx, absint(incx), 0, nl-1, &reset, 0.5)
								if nl > 1 {
									x.Set(nl/2-1, 0)
									xx.Set(absint(incx)*(nl/2-1), 0)
								}

								for _, incy = range inc {
									ly := absint(incy) * ml

									for _, alpha = range alf {

										for _, beta = range bet {
											//                             Generate the vector Y.
											y = vf(ml)
											yy = vf(ly)
											dmakeL2V(1, ml, y, 1, yy, absint(incy), 0, ml-1, &reset, 0.0)

											nc++

											//                             Save every datum before calling the
											//                             subroutine.
											transs := trans
											ms = m
											ns = n
											kls = kl
											kus = ku
											als = alpha
											as = aa.DeepCopy()
											ldas = lda
											xs := xx.DeepCopy()
											incxs = incx
											bls = beta
											ys := yy.DeepCopy()
											incys = incy

											//                             Call the subroutine.
											if full {
												Dgemv(trans, &m, &n, &alpha, aa, &lda, xx, &incx, &beta, yy, &incy)
											} else if banded {
												Dgbmv(trans, &m, &n, &kl, &ku, &alpha, aa, &lda, xx, &incx, &beta, yy, &incy)
											}

											//                             Check if error-exit was taken incorrectly.
											if !ok {
												t.Fail()
												fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
												fatal = true
												goto label130
											}

											//                             See what data changed inside subroutines.
											isame[0] = trans == transs
											isame[1] = ms == m
											isame[2] = ns == n
											if full {
												isame[3] = als == alpha
												// isame[4] = reflect.DeepEqual(*aa, *as)
												isame[5] = ldas == lda
												isame[6] = reflect.DeepEqual(*xx, *xs)
												isame[7] = incxs == incx
												isame[8] = bls == beta
												if null {
													isame[9] = reflect.DeepEqual(*yy, *ys)
												} else {
													isame[9] = lderesV(mat.General, mat.Full, 1, ml, ys, yy, absint(incy))
												}
												isame[10] = incys == incy
											} else if banded {
												isame[3] = kls == kl
												isame[4] = kus == ku
												isame[5] = als == alpha
												isame[6] = reflect.DeepEqual(aa, as)
												isame[7] = ldas == lda
												isame[8] = reflect.DeepEqual(xx, xs)
												isame[9] = incxs == incx
												isame[10] = bls == beta
												if null {
													isame[11] = reflect.DeepEqual(yy, ys)
												} else {
													isame[11] = lderesV(mat.General, mat.Full, 1, ml, ys, yy, absint(incy))
												}
												isame[12] = incys == incy
											}

											//                             If data was incorrectly changed, report
											//                             and return.
											same = true
											for i = 1; i <= nargs; i++ {
												same = same && isame[i-1]
												if !isame[i-1] {
													t.Fail()
													fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
												}
											}
											if !same {
												fatal = true
												goto label130
											}

											if !null {
												//                                Check the result.
												dmvch(trans, m, n, alpha, a, nmax, x, incx, beta, y, incy, yt, g, yy, eps, &err, &fatal, true, t)
												errmax = maxf64(errmax, err)
												//                                If got really bad answer, report and
												//                                return.
												if fatal {
													goto label130
												}
											} else {
												//                                Avoid repeating tests with M.le.0 or
												//                                N.le.0.
												goto label110
											}
										}
									}

								}
							}

						}

					}

				label110:
				}

			}

			//     Report result.
			if errmax < thresh {
				if full {
					passL2(sname, nc, 3460, t)
				} else if banded {
					passL2(sname, nc, 13828, t)
				}
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label130:
			if full {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%3d,%4.1f, A,%3d, X,%3d,%4.1f, Y,%3d)         .\n", sname, nc, sname, trans.String(), m, n, alpha, lda, incx, beta, incy)
			} else if banded {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%3d,%3d,%3d,%4.1f, A,%3d, X,%3d,%4.1f, Y,%3d) .\n", sname, nc, sname, trans.String(), m, n, kl, ku, alpha, lda, incx, beta, incy)
			}

		} else if sname == "Dsymv" || sname == "Dsbmv" || sname == "Dspmv" {
			var i, ik, incx, incxs, incy, incys, k, ks, lda, ldas, lx, ly, n, nargs, nc, nk, ns int
			var full bool = sname[2] == 'y'
			var banded bool = sname[2] == 'b'
			var packed bool = sname[2] == 'p'

			//     Define the number of arguments.
			opts.Style = mat.Symmetric
			optsfull.Style = mat.Symmetric
			if full {
				nargs = 10
				opts.Storage = mat.Dense
			} else if banded {
				nargs = 11
				opts.Storage = mat.Banded
			} else if packed {
				nargs = 9
				opts.Storage = mat.Packed
			}

			nc = 0

			for _, n = range idim {
				if banded {
					nk = nkb
				} else {
					nk = 1
				}

				for ik = 1; ik <= nk; ik++ {
					if banded {
						k = kb[ik-1]
					} else {
						k = n - 1
					}
					opts.Kl, opts.Ku = k, k
					//           Set LDA to 1 more than minimum value if room.
					if banded {
						lda = k + 1
					} else {
						lda = n
					}
					if lda < nmax {
						lda++
					}
					//           Skip tests if not enough room.
					if lda > nmax {
						goto label1100
					}
					// if packed {
					// 	laa = (n * (n + 1)) / 2
					// } else {
					// 	laa = lda * n
					// }
					null = n <= 0

					for _, uplo = range ichu {
						opts.Uplo = uplo
						optsfull.Uplo = uplo
						//              Generate the matrix A.
						a = mf(nmax, nmax, optsfull)
						aa = mf(lda, n, opts)
						dmakeL2M(n, n, a, nmax, aa, lda, k, k, &reset, 0.0)

						for _, incx = range inc {
							lx = absint(incx) * n

							//                 Generate the vector X.
							x = vf(n)
							xx = vf(lx)
							dmakeL2V(1, n, x, 1, xx, absint(incx), 0, n-1, &reset, 0.5)
							if n > 1 {
								x.Set(n/2-1, 0)
								xx.Set(absint(incx)*(n/2-1), 0)
							}

							for _, incy = range inc {
								ly = absint(incy) * n

								for _, alpha = range alf {

									for _, beta = range bet {
										//                          Generate the vector Y.
										y = vf(n)
										yy = vf(ly)
										dmakeL2V(1, n, y, 1, yy, absint(incy), 0, n-1, &reset, 0.0)

										nc++

										//                          Save every datum before calling the
										//                          subroutine.
										uplos = uplo
										ns = n
										ks = k
										als = alpha
										as = aa.DeepCopy()
										ldas = lda
										xs := xx.DeepCopy()
										incxs = incx
										bls = beta
										ys := yy.DeepCopy()
										incys = incy
										//
										//                          Call the subroutine.
										//
										if full {
											Dsymv(uplo, &n, &alpha, aa, &lda, xx, &incx, &beta, yy, &incy)
										} else if banded {
											Dsbmv(uplo, &n, &k, &alpha, aa, &lda, xx, &incx, &beta, yy, &incy)
										} else if packed {
											Dspmv(uplo, &n, &alpha, aa.VectorIdx(0), xx, &incx, &beta, yy, &incy)
										}
										//
										//                          Check if error-exit was taken incorrectly.
										//
										if !ok {
											t.Fail()
											fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
											fatal = true
											goto label1120
										}
										//
										//                          See what data changed inside subroutines.
										//
										isame[0] = uplo == uplos
										isame[1] = ns == n
										if full {
											isame[2] = als == alpha
											isame[3] = reflect.DeepEqual(aa, as)
											isame[4] = ldas == lda
											isame[5] = reflect.DeepEqual(xx, xs)
											isame[6] = incxs == incx
											isame[7] = bls == beta
											if null {
												isame[8] = reflect.DeepEqual(yy, ys)
											} else {
												isame[8] = lderesV(mat.General, mat.Full, 1, n, ys, yy, absint(incy))
											}
											isame[9] = incys == incy
										} else if banded {
											isame[2] = ks == k
											isame[3] = als == alpha
											isame[4] = reflect.DeepEqual(aa, as)
											isame[5] = ldas == lda
											isame[6] = reflect.DeepEqual(xx, xs)
											isame[7] = incxs == incx
											isame[8] = bls == beta
											if null {
												isame[9] = reflect.DeepEqual(yy, ys)
											} else {
												isame[9] = lderesV(mat.General, mat.Full, 1, n, ys, yy, absint(incy))
											}
											isame[10] = incys == incy
										} else if packed {
											isame[2] = als == alpha
											isame[3] = reflect.DeepEqual(aa, as)
											isame[4] = reflect.DeepEqual(xx, xs)
											isame[5] = incxs == incx
											isame[6] = bls == beta
											if null {
												isame[7] = reflect.DeepEqual(yy, ys)
											} else {
												isame[7] = lderesV(mat.General, mat.Full, 1, n, ys, yy, absint(incy))
											}
											isame[8] = incys == incy
										}

										//                          If data was incorrectly changed, report and
										//                          return.
										same = true
										for i = 1; i <= nargs; i++ {
											same = same && isame[i-1]
											if !isame[i-1] {
												t.Fail()
												fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
											}
										}
										if !same {
											fatal = true
											goto label1120
										}

										if !null {
											//                             Check the result.
											dmvch(mat.NoTrans, n, n, alpha, a, nmax, x, incx, beta, y, incy, yt, g, yy, eps, &err, &fatal, true, t)

											errmax = maxf64(errmax, err)

											//                             If got really bad answer, report and
											//                             return.
											if fatal {
												goto label1120
											}
										} else {
											//                             Avoid repeating tests with N.le.0
											goto label1110
										}

									}

								}
							}
						}
					}

				label1100:
				}

			label1110:
			}

			//     Report result.
			if errmax < thresh {
				if full || packed {
					passL2(sname, nc, 1441, t)
				} else if banded {
					passL2(sname, nc, 5761, t)
				}
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label1120:
			;
			if full {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%4.1f, A,%3d, X,%3d,%4.1f, Y,%3d)             .\n", sname, nc, sname, uplo.String(), n, alpha, lda, incx, beta, incy)
			} else if banded {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%3d,%4.1f, A,%3d, X,%3d,%4.1f, Y,%3d)         .\n", sname, nc, sname, uplo.String(), n, k, alpha, lda, incx, beta, incy)
			} else if packed {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%4.1f, AP, X,%3d,%4.1f, Y,%3d)                .\n", sname, nc, sname, uplo.String(), n, alpha, incx, beta, incy)
			}

		} else if sname == "Dtrmv" || sname == "Dtbmv" || sname == "Dtpmv" || sname == "Dtrsv" || sname == "Dtbsv" || sname == "Dtpsv" {
			var i, ik, incx, incxs, k, ks, lda, ldas, lx, n, nargs, nc, nk, ns int
			var full bool = sname[2] == 'r'
			var banded bool = sname[2] == 'b'
			var packed bool = sname[2] == 'p'

			//     Define the number of arguments.
			opts.Style = mat.Triangular
			optsfull.Style = mat.Triangular
			if full {
				nargs = 8
				opts.Storage = mat.Dense
			} else if banded {
				nargs = 9
				opts.Storage = mat.Banded
			} else if packed {
				nargs = 7
				opts.Storage = mat.Packed
			}

			nc = 0

			for _, n = range idim {
				if banded {
					nk = nkb
				} else {
					nk = 1
				}

				for ik = 1; ik <= nk; ik++ {
					if banded {
						k = kb[ik-1]
					} else {
						k = n - 1
					}
					opts.Kl, opts.Ku = k, k
					//           Set LDA to 1 more than minimum value if room.
					if banded {
						lda = k + 1
					} else {
						lda = n
					}
					if lda < nmax {
						lda++
					}
					//           Skip tests if not enough room.
					if lda > nmax {
						goto label2100
					}
					// if packed {
					// 	laa = (n * (n + 1)) / 2
					// } else {
					// 	laa = lda * n
					// }
					null = n <= 0

					for _, uplo = range ichu {
						opts.Uplo = uplo
						optsfull.Uplo = uplo

						for _, trans = range icht {

							for _, diag = range ichd {
								opts.Diag = diag
								optsfull.Diag = diag

								//                    Generate the matrix A.
								a = mf(nmax, nmax, optsfull)
								aa = mf(nmax, nmax, opts)
								dmakeL2M(n, n, a, nmax, aa, lda, k, k, &reset, 0.0)

								for _, incx = range inc {
									lx = absint(incx) * n

									//                       Generate the vector X.
									x = vf(n)
									xx = vf(lx)
									dmakeL2V(1, n, x, 1, xx, absint(incx), 0, n-1, &reset, 0.5)
									if n > 1 {
										x.Set(n/2-1, 0)
										xx.Set(absint(incx)*(n/2-1), 0)
									}

									nc++

									//                       Save every datum before calling the subroutine.
									uplos = uplo
									transs = trans
									diags = diag
									ns = n
									ks = k
									as = aa.DeepCopy()
									ldas = lda
									xs := xx.DeepCopy()
									incxs = incx

									//                       Call the subroutine.
									if sname[3:5] == "mv" {
										if full {
											Dtrmv(uplo, trans, diag, &n, aa, &lda, xx, &incx)
										} else if banded {
											Dtbmv(uplo, trans, diag, &n, &k, aa, &lda, xx, &incx)
										} else if packed {
											Dtpmv(uplo, trans, diag, &n, aa.VectorIdx(0), xx, &incx)
										}
									} else if sname[3:5] == "sv" {
										if full {
											Dtrsv(uplo, trans, diag, &n, aa, &lda, xx, &incx)
										} else if banded {
											Dtbsv(uplo, trans, diag, &n, &k, aa, &lda, xx, &incx)
										} else if packed {
											Dtpsv(uplo, trans, diag, &n, aa.VectorIdx(0), xx, &incx)
										}
									}

									//                       Check if error-exit was taken incorrectly.
									if !ok {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
										fatal = true
										goto label2120
									}

									//                       See what data changed inside subroutines.
									isame[0] = uplo == uplos
									isame[1] = trans == transs
									isame[2] = diag == diags
									isame[3] = ns == n
									if full {
										isame[4] = reflect.DeepEqual(aa, as)
										isame[5] = ldas == lda
										if null {
											isame[6] = reflect.DeepEqual(xx, xs)
										} else {
											isame[6] = lderesV(mat.General, mat.Full, 1, n, xs, xx, absint(incx))
										}
										isame[7] = incxs == incx
									} else if banded {
										isame[4] = ks == k
										isame[5] = reflect.DeepEqual(aa, as)
										isame[6] = ldas == lda
										if null {
											isame[7] = reflect.DeepEqual(xx, xs)
										} else {
											isame[7] = lderesV(mat.General, mat.Full, 1, n, xs, xx, absint(incx))
										}
										isame[8] = incxs == incx
									} else if packed {
										isame[4] = reflect.DeepEqual(aa, as)
										if null {
											isame[5] = reflect.DeepEqual(xx, xs)
										} else {
											isame[5] = lderesV(mat.General, mat.Full, 1, n, xs, xx, absint(incx))
										}
										isame[6] = incxs == incx
									}

									//                       If data was incorrectly changed, report and
									//                       return.
									same = true
									for i = 1; i <= nargs; i++ {
										same = same && isame[i-1]
										if !isame[i-1] {
											t.Fail()
											fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
										}
									}
									if !same {
										fatal = true
										goto label2120
									}

									if !null {
										z := x.DeepCopy()
										if sname[3:5] == "mv" {
											//                             Check the result.
											dmvch(trans, n, n, 1.0, a, nmax, x, incx, 0.0, z, incx, xt, g, xx, eps, &err, &fatal, true, t)
										} else if sname[3:5] == "sv" {
											//                             Compute approximation to original vector.
											for i = 1; i <= n; i++ {
												z.Set(i-1, xx.Get((i-1)*absint(incx)))
												xx.Set((i-1)*absint(incx), x.Get(i-1))
											}
											dmvch(trans, n, n, 1.0, a, nmax, z, incx, 0.0, x, incx, xt, g, xx, eps, &err, &fatal, true, t)
										}
										errmax = maxf64(errmax, err)
										//                          If got really bad answer, report and return.
										if fatal {
											goto label2120
										}
									} else {
										//                          Avoid repeating tests with N.le.0.
										goto label2110
									}

								}
							}

						}

					}

				label2100:
				}

			label2110:
			}

			//     Report result.
			if errmax < thresh {
				if full || packed {
					passL2(sname, nc, 241, t)
				} else if banded {
					passL2(sname, nc, 961, t)
				}
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label2120:
			;
			if full {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%v,%v,%3d, A,%3d, X,%3d)                     .\n", sname, nc, sname, uplo.String(), trans.String(), diag.String(), n, lda, incx)
			} else if banded {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%v,%v,%3d,%3d, A,%3d, X,%3d)                 .\n", sname, nc, sname, uplo.String(), trans.String(), diag.String(), n, k, lda, incx)
			} else if packed {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%v,%v,%3d, AP, X,%3d)                        .\n", sname, nc, sname, uplo.String(), trans.String(), diag.String(), n, incx)
			}

		} else if sname == "Dger" {
			var i, im, incx, incxs, incy, incys, j, lda, ldas, lx, ly, m, ms, n, nd, ns int
			var nargs int = 9
			var nc int = 0

			opts.Style = mat.General
			opts.Storage = mat.Dense
			optsfull.Style = mat.General

			for i := range isame {
				isame[i] = true
			}

			for _, n = range idim {
				nd = n/2 + 1

				for im = 1; im <= 2; im++ {
					if im == 1 {
						m = maxint(n-nd, 0)
					}
					if im == 2 {
						m = minint(n+nd, nmax)
					}
					//
					//           Set LDA to 1 more than minimum value if room.
					lda = m
					if lda < nmax {
						lda++
					}
					//           Skip tests if not enough room.
					if lda > nmax {
						continue
					}
					// laa = lda * n
					null = n <= 0 || m <= 0

					for _, incx = range inc {
						lx = absint(incx) * m

						//              Generate the vector X.
						x = vf(m)
						xx = vf(lx)
						dmakeL2V(1, m, x, 1, xx, absint(incx), 0, m-1, &reset, 0.5)
						if m > 1 {
							x.Set(m/2-1, 0)
							xx.Set(absint(incx)*(m/2-1), 0)
						}

						for _, incy = range inc {
							ly = absint(incy) * n

							//                 Generate the vector Y.
							y = vf(n)
							yy = vf(ly)
							dmakeL2V(1, n, y, 1, yy, absint(incy), 0, n-1, &reset, 0.0)
							if n > 1 {
								y.Set(n/2-1, 0)
								yy.Set(absint(incy)*(n/2-1), 0)
							}

							for _, alpha = range alf {
								//                    Generate the matrix A.
								a = mf(nmax, nmax, optsfull)
								aa = mf(lda, n, opts)
								dmakeL2M(m, n, a, nmax, aa, lda, m-1, n-1, &reset, 0.0)

								nc++

								//                    Save every datum before calling the subroutine.
								ms = m
								ns = n
								als = alpha
								as = aa.DeepCopy()
								ldas = lda
								xs = xx.DeepCopy()
								incxs = incx
								ys = yy.DeepCopy()
								incys = incy

								//                    Call the subroutine.
								Dger(&m, &n, &alpha, xx, &incx, yy, &incy, aa, &lda)

								//                    Check if error-exit was taken incorrectly.
								if !ok {
									t.Fail()
									fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
									fatal = true
									goto label3140
								}

								//                    See what data changed inside subroutine.
								isame[0] = ms == m
								isame[1] = ns == n
								isame[2] = als == alpha
								isame[3] = reflect.DeepEqual(xx, xs)
								isame[4] = ldas == lda
								isame[5] = reflect.DeepEqual(yy, ys)
								isame[6] = incxs == incx
								if null {
									isame[7] = reflect.DeepEqual(aa, as)
								} else {
									isame[7] = lderesM(m, n, as, aa, lda)
								}
								isame[8] = incys == incy

								//                    If data was incorrectly changed, report and return.
								same = true
								for i = 1; i <= nargs; i++ {
									same = same && isame[i-1]
									if !isame[i-1] {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
									}
								}
								if !same {
									fatal = true
									goto label3140
								}

								if !null {
									//                       Check the result column by column.
									z := mf(m, nmax, opts)
									w := y.DeepCopy()
									if incx > 0 {
										for i = 1; i <= m; i++ {
											z.Set(i-1, 0, x.Get(i-1))
										}
									} else {
										for i = 1; i <= m; i++ {
											z.Set(i-1, 0, x.Get(m-i+1-1))
										}
									}
									for j = 1; j <= n; j++ {
										if incy > 0 {
											w.Set(0, y.Get(j-1))
										} else {
											w.Set(0, y.Get(n-j+1-1))
										}
										dmvch(mat.NoTrans, m, 1, alpha, z, nmax, w, 1, 1.0, a.Vector(0, j-1), 1, yt, g, aa.Vector(0, j-1), eps, &err, &fatal, true, t)
										errmax = maxf64(errmax, err)
										//                          If got really bad answer, report and return.
										if fatal {
											goto label3130
										}

									}
								} else {
									//                       Avoid repeating tests with M.le.0 or N.le.0.
									goto label3110
								}

							}
						}
					}

				label3110:
				}

			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 388, t)
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label3130:
			;
			fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)

		label3140:
			;
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%3d,%3d,%4.1f, X,%3d, Y,%3d, A,%3d)                  .\n", sname, nc, sname, m, n, alpha, incy, incx, lda)

		} else if sname == "Dsyr" || sname == "Dspr" {
			var i, incx, incxs, j, ja, jj, lda, ldas, lj, lx, n, nargs, nc, ns int
			var full bool = sname[2] == 'y'
			var packed bool = sname[2] == 'p'

			//     Define the number of arguments.
			opts.Style = mat.Symmetric
			optsfull.Style = mat.Symmetric
			if full {
				nargs = 7
				opts.Storage = mat.Dense
			} else if packed {
				nargs = 6
				opts.Storage = mat.Packed
			}

			nc = 0

			for _, n = range idim {
				//        Set LDA to 1 more than minimum value if room.
				lda = n
				if lda < nmax {
					lda++
				}
				//        Skip tests if not enough room.
				if lda > nmax {
					goto label4100
				}
				// if packed {
				// 	laa = (n * (n + 1)) / 2
				// } else {
				// 	laa = lda * n
				// }

				for _, uplo = range ichu {
					upper = uplo == mat.Upper
					opts.Uplo = uplo
					optsfull.Uplo = uplo

					for _, incx = range inc {
						lx = absint(incx) * n

						//              Generate the vector X.
						x = vf(n)
						xx = vf(lx)
						dmakeL2V(1, n, x, 1, xx, absint(incx), 0, n-1, &reset, 0.5)
						if n > 1 {
							x.Set(n/2-1, 0)
							xx.Set(absint(incx)*(n/2-1), 0)
						}

						for _, alpha = range alf {
							null = n <= 0 || alpha == 0

							//                 Generate the matrix A.
							a = mf(nmax, nmax, optsfull)
							aa = mf(lda, n, opts)

							nc++

							//                 Save every datum before calling the subroutine.
							uplos = uplo
							ns = n
							als = alpha
							as = aa.DeepCopy()
							ldas = lda
							xs := xx.DeepCopy()
							incxs = incx

							//                 Call the subroutine.
							if full {
								Dsyr(uplo, &n, &alpha, xx, &incx, aa, &lda)
							} else if packed {
								Dspr(uplo, &n, &alpha, xx, &incx, aa.VectorIdx(0))
							}

							//                 Check if error-exit was taken incorrectly.
							if !ok {
								t.Fail()
								fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
								fatal = true
								goto label4120
							}

							//                 See what data changed inside subroutines.
							isame[0] = uplo == uplos
							isame[1] = ns == n
							isame[2] = als == alpha
							isame[3] = reflect.DeepEqual(xx, xs)
							isame[4] = incxs == incx
							if null {
								isame[5] = reflect.DeepEqual(aa, as)
							} else {
								isame[5] = lderesM(n, n, as, aa, lda)
							}
							if !packed {
								isame[6] = ldas == lda
							}

							//                 If data was incorrectly changed, report and return.
							same = true
							for i = 1; i <= nargs; i++ {
								same = same && isame[i-1]
								if !isame[i-1] {
									t.Fail()
									fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
								}
							}
							if !same {
								fatal = true
								goto label4120
							}

							if !null {
								//                    Check the result column by column.
								z := mf(n, 1, optsge)
								w := vf(n)
								if incx > 0 {
									for i = 1; i <= n; i++ {
										z.Set(i-1, 0, x.Get(i-1))
									}
								} else {
									for i = 1; i <= n; i++ {
										z.Set(i-1, 0, x.Get(n-i+1-1))
									}
								}
								ja = 1
								for j = 1; j <= n; j++ {
									w.Set(0, z.Get(j-1, 0))
									if upper {
										jj = 1
										lj = j
									} else {
										jj = j
										lj = n - j + 1
									}
									dmvch(mat.NoTrans, lj, 1, alpha, z.OffIdx(jj-1), lj, w, 1, 1.0, a.Vector(jj-1, j-1), 1, yt, g, aa.VectorIdx(ja-1), eps, &err, &fatal, true, t)
									if full {
										if upper {
											ja += lda
										} else {
											ja += lda + 1
										}
									} else {
										ja += lj
									}
									errmax = maxf64(errmax, err)
									//                       If got really bad answer, report and return.
									if fatal {
										goto label4110
									}
								}
							} else {
								//                    Avoid repeating tests if N.le.0.
								if n <= 0 {
									goto label4100
								}
							}

						}
					}

				}

			label4100:
			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 121, t)
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label4110:
			;
			fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)

		label4120:
			;
			if full {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%4.1f, X,%3d, A,%3d)                        .\n", sname, nc, sname, uplo.String(), n, alpha, incx, lda)
			} else if packed {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%4.1f, X,%3d, AP)                           .\n", sname, nc, sname, uplo.String(), n, alpha, incx)
			}

		} else if sname == "Dsyr2" || sname == "Dspr2" {
			var i, incx, incxs, incy, incys, j, ja, jj, lda, ldas, lj, lx, ly, n, nargs, nc, ns int
			var full = sname[2] == 'y'
			var packed = sname[2] == 'p'

			//     Define the number of arguments.
			opts.Style = mat.Symmetric
			optsfull.Style = mat.Symmetric
			if full {
				nargs = 9
				opts.Storage = mat.Dense
			} else if packed {
				nargs = 8
				opts.Storage = mat.Packed
			}

			nc = 0

			for _, n = range idim {
				//        Set LDA to 1 more than minimum value if room.
				lda = n
				if lda < nmax {
					lda++
				}
				//        Skip tests if not enough room.
				if lda > nmax {
					goto label5140
				}
				// if packed {
				// 	laa = (n * (n + 1)) / 2
				// } else {
				// 	laa = lda * n
				// }

				for _, uplo = range ichu {
					upper = uplo == mat.Upper
					opts.Uplo = uplo
					optsfull.Uplo = uplo

					for _, incx = range inc {
						lx = absint(incx) * n

						//              Generate the vector X.
						x = vf(n)
						xx = vf(lx)
						dmakeL2V(1, n, x, 1, xx, absint(incx), 0, n-1, &reset, 0.5)
						if n > 1 {
							x.Set(n/2-1, 0)
							xx.Set(absint(incx)*(n/2-1), 0)
						}

						for _, incy = range inc {
							ly = absint(incy) * n

							//                 Generate the vector Y.
							y = vf(n)
							yy = vf(ly)
							dmakeL2V(1, n, y, 1, yy, absint(incy), 0, n-1, &reset, 0.0)
							if n > 1 {
								y.Set(n/2-1, 0)
								yy.Set(absint(incy)*(n/2-1), 0)
							}

							for _, alpha = range alf {
								null = n <= 0 || alpha == 0

								//                    Generate the matrix A.
								a = mf(nmax, nmax, optsfull)
								aa = mf(lda, n, opts)

								nc++

								//                    Save every datum before calling the subroutine.
								uplos = uplo
								ns = n
								als = alpha
								as = aa.DeepCopy()
								ldas = lda
								xs = xx.DeepCopy()
								incxs = incx
								ys = yy.DeepCopy()
								incys = incy

								//                    Call the subroutine.
								if full {
									Dsyr2(uplo, &n, &alpha, xx, &incx, yy, &incy, aa, &lda)
								} else if packed {
									Dspr2(uplo, &n, &alpha, xx, &incx, yy, &incy, aa.VectorIdx(0))
								}

								//                    Check if error-exit was taken incorrectly.
								if !ok {
									t.Fail()
									fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
									fatal = true
									goto label5160
								}

								//                    See what data changed inside subroutines.
								isame[0] = uplo == uplos
								isame[1] = ns == n
								isame[2] = als == alpha
								isame[3] = reflect.DeepEqual(xx, xs)
								isame[4] = incxs == incx
								isame[5] = reflect.DeepEqual(yy, ys)
								isame[6] = incys == incy
								if null {
									isame[7] = reflect.DeepEqual(aa, as)
								} else {
									isame[7] = lderesM(n, n, as, aa, lda)
								}
								if !packed {
									isame[8] = ldas == lda
								}

								//                    If data was incorrectly changed, report and return.
								same = true
								for i = 1; i <= nargs; i++ {
									same = same && isame[i-1]
									if !isame[i-1] {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
									}
								}
								if !same {
									fatal = true
									goto label5160
								}

								if !null {
									//                    Check the result column by column.
									z := mf(n, 2, optsge)
									w := vf(2)
									if incx > 0 {
										for i = 1; i <= n; i++ {
											z.Set(i-1, 0, x.Get(i-1))
										}
									} else {
										for i = 1; i <= n; i++ {
											z.Set(i-1, 0, x.Get(n-i+1-1))
										}
									}
									if incy > 0 {
										for i = 1; i <= n; i++ {
											z.Set(i-1, 1, y.Get(i-1))
										}
									} else {
										for i = 1; i <= n; i++ {
											z.Set(i-1, 1, y.Get(n-i+1-1))
										}
									}
									ja = 1
									for j = 1; j <= n; j++ {
										w.Set(0, z.Get(j-1, 1))
										w.Set(1, z.Get(j-1, 0))
										if upper {
											jj = 1
											lj = j
										} else {
											jj = j
											lj = n - j + 1
										}
										dmvch(mat.NoTrans, lj, 2, alpha, z.OffIdx(jj-1), nmax, w, 1, 1.0, a.Vector(jj-1, j-1), 1, yt, g, aa.VectorIdx(ja-1), eps, &err, &fatal, true, t)
										if full {
											if upper {
												ja += lda
											} else {
												ja += lda + 1
											}
										} else {
											ja += lj
										}
										errmax = maxf64(errmax, err)
										//                          If got really bad answer, report and return.
										if fatal {
											goto label5150
										}
									}
								} else {
									//                       Avoid repeating tests with N.le.0.
									if n <= 0 {
										goto label5140
									}
								}

							}
						}
					}

				}

			label5140:
			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 481, t)
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label5150:
			;
			fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)

		label5160:

			if full {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%4.1f, X,%3d, Y,%3d, A,%3d)                  .\n", sname, nc, sname, uplo.String(), n, alpha, incx, incy, lda)
			} else if packed {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%4.1f, X,%3d, Y,%3d, AP)                     .\n", sname, nc, sname, uplo.String(), n, alpha, incx, incy)
			}
		}
	}
}

func dmvch(trans mat.MatTrans, m, n int, alpha float64, a *mat.Matrix, nmax int, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int, yt, g, yy *mat.Vector, eps float64, err *float64, fatal *bool, mv bool, t *testing.T) {
	var tran bool
	var erri, one, zero float64
	var i, incxl, incyl, iy, j, jx, kx, ky, ml, nl int

	zero = 0.0
	one = 1.0

	tran = trans.IsTrans()
	if tran {
		ml = n
		nl = m
	} else {
		ml = m
		nl = n
	}
	if incx < 0 {
		kx = nl
		incxl = -1
	} else {
		kx = 1
		incxl = 1
	}
	if incy < 0 {
		ky = ml
		incyl = -1
	} else {
		ky = 1
		incyl = 1
	}

	//     Compute expected result in YT using data in A, X and Y.
	//     Compute gauges in G.
	iy = ky
	for i = 1; i <= ml; i++ {
		yt.Set(iy-1, zero)
		g.Set(iy-1, zero)
		jx = kx
		if tran {
			for j = 1; j <= nl; j++ {
				yt.Set(iy-1, yt.Get(iy-1)+a.Get(j-1, i-1)*x.Get(jx-1))
				g.Set(iy-1, g.Get(iy-1)+math.Abs(a.Get(j-1, i-1)*x.Get(jx-1)))
				jx += incxl
			}
		} else {
			for j = 1; j <= nl; j++ {
				yt.Set(iy-1, yt.Get(iy-1)+a.Get(i-1, j-1)*x.Get(jx-1))
				g.Set(iy-1, g.Get(iy-1)+math.Abs(a.Get(i-1, j-1)*x.Get(jx-1)))
				jx += incxl
			}
		}
		yt.Set(iy-1, alpha*yt.Get(iy-1)+beta*y.Get(iy-1))
		g.Set(iy-1, math.Abs(alpha)*g.Get(iy-1)+math.Abs(beta*y.Get(iy-1)))
		iy += incyl
	}

	//     Compute the error ratio for this result.
	(*err) = zero
	for i = 1; i <= ml; i++ {
		erri = math.Abs(yt.Get(i-1)-yy.Get((i-1)*absint(incy))) / eps
		if g.Get(i-1) != zero {
			erri /= g.Get(i - 1)
		}
		(*err) = maxf64(*err, erri)
		if (*err)*math.Sqrt(eps) >= one {
			goto label50
		}
	}
	//     If the loop completes, all results are at least half accurate.
	goto label70

	//     Report fatal error.
label50:
	;
	(*fatal) = true
	fmt.Printf(" ******* FATAL ERROR - COMPUTED RESULT IS LESS THAN HALF ACCURATE *******\n           EXPECTED RESULT   COMPUTED RESULT\n")
	for i = 0; i < ml; i++ {
		if mv {
			fmt.Printf(" %7d%18.6f%18.6f\n", i, yt.Get(i), yy.Get(i))
		} else {
			fmt.Printf(" %7d%18.6f%18.6f\n", i, yy.Get(i), yt.Get(i))
		}
	}

label70:
}

func dmakeL2V(m, n int, a *mat.Vector, nmax int, aa *mat.Vector, lda, kl, ku int, reset *bool, transl float64) {
	var rogue, zero float64
	var i, j int

	zero = 0.0
	rogue = -1.0e10

	//     Generate data in array A.
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			if (i <= j && j-i <= ku) || (i >= j && i-j <= kl) {
				a.Set(i-1+(j-1)*nmax, dbeg(reset)+transl)
			} else {
				a.Set(i-1+(j-1)*nmax, zero)
			}
		}
	}

	//     Store elements in array AS in data structure required by routine.
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			aa.Set(i-1+(j-1)*lda, a.Get(i-1+(j-1)*nmax))
		}
		for i = m + 1; i <= lda; i++ {
			aa.Set(i-1+(j-1)*lda, rogue)
		}
	}
}

func dmakeL2M(m, n int, a *mat.Matrix, nmax int, aa *mat.Matrix, lda, kl, ku int, reset *bool, transl float64) {
	var one, rogue, zero float64
	var i, i1, i2, i3, ibeg, iend, ioff, j, kk int

	one = 1.0
	zero = 0.0
	rogue = -1.0e10

	gen := aa.Opts.Style == mat.General
	sym := aa.Opts.Style == mat.Symmetric
	tri := aa.Opts.Style == mat.Triangular
	upper := (sym || tri) && aa.Opts.Uplo == mat.Upper
	lower := (sym || tri) && aa.Opts.Uplo == mat.Lower
	unit := tri && aa.Opts.Diag == mat.Unit
	dense := aa.Opts.Storage == mat.Dense
	banded := aa.Opts.Storage == mat.Banded
	packed := aa.Opts.Storage == mat.Packed

	//     Generate data in array A.
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			if gen || (upper && i <= j) || (lower && i >= j) {
				if (i <= j && j-i <= ku) || (i >= j && i-j <= kl) {
					a.Set(i-1, j-1, dbeg(reset)+transl)
				} else {
					a.Set(i-1, j-1, zero)
				}
				if i != j {
					if sym {
						a.Set(j-1, i-1, a.Get(i-1, j-1))
					} else if tri {
						a.Set(j-1, i-1, zero)
					}
				}
			}
		}
		if tri {
			a.Set(j-1, j-1, a.Get(j-1, j-1)+one)
		}
		if unit {
			a.Set(j-1, j-1, one)
		}
	}

	//     Store elements in array AS in data structure required by routine.
	if gen && dense {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				aa.Set(i-1, j-1, a.Get(i-1, j-1))
			}
			for i = m + 1; i <= lda; i++ {
				aa.Set(i-1, j-1, rogue)
			}
		}
	} else if gen && banded {
		for j = 1; j <= n; j++ {
			for i1 = 1; i1 <= ku+1-j; i1++ {
				aa.Set(i1-1, j-1, rogue)
			}
			for i2 = i1; i2 <= minint(kl+ku+1, ku+1+m-j); i2++ {
				aa.Set(i2-1, j-1, a.Get(i2+j-ku-1-1, j-1))
			}
			for i3 = i2; i3 <= lda; i3++ {
				aa.Set(i3-1, j-1, rogue)
			}
		}
	} else if (sym || tri) && dense {
		for j = 1; j <= n; j++ {
			if upper {
				ibeg = 1
				if unit {
					iend = j - 1
				} else {
					iend = j
				}
			} else {
				if unit {
					ibeg = j + 1
				} else {
					ibeg = j
				}
				iend = n
			}
			for i = 1; i <= ibeg-1; i++ {
				aa.Set(i-1, j-1, rogue)
			}
			for i = ibeg; i <= iend; i++ {
				aa.Set(i-1, j-1, a.Get(i-1, j-1))
			}
			for i = iend + 1; i <= lda; i++ {
				aa.Set(i-1, j-1, rogue)
			}
		}
	} else if (sym || tri) && banded {
		for j = 1; j <= n; j++ {
			if upper {
				kk = kl + 1
				ibeg = maxint(1, kl+2-j)
				if unit {
					iend = kl
				} else {
					iend = kl + 1
				}
			} else {
				kk = 1
				if unit {
					ibeg = 2
				} else {
					ibeg = 1
				}
				iend = minint(kl+1, 1+m-j)
			}
			for i = 1; i <= ibeg-1; i++ {
				aa.Set(i-1, j-1, rogue)
			}
			for i = ibeg; i <= iend; i++ {
				aa.Set(i-1, j-1, a.Get(i+j-kk-1, j-1))
			}
			for i = iend + 1; i <= lda; i++ {
				aa.Set(i-1, j-1, rogue)
			}
		}
	} else if (sym || tri) && packed {
		ioff = 0
		for j = 1; j <= n; j++ {
			if upper {
				ibeg = 1
				iend = j
			} else {
				ibeg = j
				iend = n
			}
			for i = ibeg; i <= iend; i++ {
				ioff = ioff + 1
				aa.Set(ioff-1, 0, a.Get(i-1, j-1))
				if i == j {
					if unit {
						aa.Set(ioff-1, 0, rogue)
					}
				}
			}
		}
	}
}

func lderesV(_type mat.MatStyle, uplo mat.MatUplo, m, n int, aa, as *mat.Vector, lda int) bool {
	var upper bool
	var i, ibeg, iend, j int

	upper = uplo == mat.Upper
	if _type == mat.General {
		for j = 1; j <= n; j++ {
			for i = m + 1; i <= lda; i++ {
				if aa.Get(i-1+(j-1)*lda) != as.Get(i-1+(j-1)*lda) {
					return false
				}
			}
		}
	} else if _type == mat.Symmetric {
		for j = 1; j <= n; j++ {
			if upper {
				ibeg = 1
				iend = j
			} else {
				ibeg = j
				iend = n
			}
			for i = 1; i <= ibeg-1; i++ {
				if aa.Get(i-1+(j-1)*lda) != as.Get(i-1+(j-1)*lda) {
					return false
				}
			}
			for i = iend + 1; i <= lda; i++ {
				if aa.Get(i-1+(j-1)*lda) != as.Get(i-1+(j-1)*lda) {
					return false
				}
			}
		}
	}

	return true
}

func lderesM(m, n int, aa, as *mat.Matrix, lda int) bool {
	var i, ibeg, iend, j int

	if aa.Opts.Style == mat.General && aa.Opts.Storage == mat.Dense {
		for j = 1; j <= n; j++ {
			for i = m + 1; i <= lda; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					return false
				}
			}
		}
	} else if aa.Opts.Style == mat.Symmetric && aa.Opts.Storage == mat.Dense {
		for j = 1; j <= n; j++ {
			if aa.Opts.Uplo == mat.Upper {
				ibeg = 1
				iend = j
			} else {
				ibeg = j
				iend = n
			}
			for i = 1; i <= ibeg-1; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					return false
				}
			}
			for i = iend + 1; i <= lda; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					return false
				}
			}
		}
	}

	return true
}

func dbeg(reset *bool) (dbegReturn float64) {
	var i *int = &common.begc.i
	var ic *int = &common.begc.ic
	var mi *int = &common.begc.mi

	if *reset {
		//        Initialize local variables.
		*mi = 891
		*i = 7
		*ic = 0
		(*reset) = false
	}
	//
	//     The sequence of values of I is bounded between 1 and 999.
	//     If initial I = 1,2,3,6,7 or 9, the period will be 50.
	//     If initial I = 4 or 8, the period will be 25.
	//     If initial I = 5, the period will be 10.
	//     IC is used to break up the period by skipping 1 value of I in 6.
	//
	*ic++
label10:
	;
	*i *= *mi
	*i -= 1000 * ((*i) / 1000)
	if *ic >= 5 {
		*ic = 0
		goto label10
	}
	dbegReturn = float64((*i)-500) / 1001.0
	return
}